import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from model.configuration import TinyuConfig
from dataset.lm_dataset import PromptDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_grpo_args
# 引入 GRPO 特有的工具函数
from trainer.grpo_utils import init_grpo_models, generate_grpo_experience, compute_grpo_advantages, gather_logprobs
from trainer.train_utils import print_model_param_details, set_seed, save_checkpoint, SkipStepSampler
from trainer.train_utils import load_checkpoint, setup_device_and_distributed, log_training_progress
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

if __name__ == "__main__":
    # ================= 0. 获取全局配置 =================
    args = parse_grpo_args()
    assert args.rollout_batch_size >= args.accumulation_steps
    # 架构签名去掉 critic 相关参数，加入 Group Size (G)
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_grpo_g{args.group_size}"

    # ================= 1. 初始化分布式环境 =================
    device, local_rank, is_distributed, is_main_process = setup_device_and_distributed()
    set_seed(args.seed + local_rank)

    # ================= 2. 初始化模型与数据 =================
    config = TinyuConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )

    # GRPO 只需要 3 个模型 (没有 Critic): 
    # 1. actor: 正在训练的策略模型
    # 2. ref_model: 参考模型，用于计算 KL 散度约束
    # 3. reward_model: 奖励模型，用于给 G 个结果打分
    actor, ref_model, reward_model, tokenizer = init_grpo_models(config, args.model_paths, device=device)

    # 冻结 Reference 和 Reward 模型
    for model in [ref_model, reward_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    if is_main_process:
        print("Actor Model Params:")
        print_model_param_details(actor, detail=False)

    # ================= 3. 模型编译与 DDP 包装 =================
    if args.use_compile:
        if is_main_process: print("正在使用 torch.compile 编译模型...")
        actor = torch.compile(actor)

    if is_distributed:
        actor._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor = DDP(actor, device_ids=[local_rank])

    # ================= 4. 数据集与优化器 =================
    dataset = PromptDataset(args.data_path, tokenizer, args.max_prompt_length)

    if is_distributed:
        base_sampler = DistributedSampler(dataset)
    else:
        from torch.utils.data import RandomSampler
        base_sampler = RandomSampler(dataset)

    base_dataloader = DataLoader(
        dataset, batch_size=args.rollout_batch_size, sampler=base_sampler, drop_last=True
    )
    full_dataloader_len = len(base_dataloader)

    # 仅保留 Actor 的优化器
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_learning_rate)

    # ================= 5. 初始化混合精度与余弦退火 =================
    ptdtype = torch.float16
    if args.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
        if is_main_process: print("硬件支持 BF16，已开启 Bfloat16 混合精度训练")
    elif args.dtype == "float32":
        ptdtype = torch.float32

    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

    total_rollouts = args.epochs * full_dataloader_len
    total_update_steps = total_rollouts * args.grpo_epochs # PPO Epoch 替换为 GRPO Epoch
    warmup_steps = int(total_update_steps * 0.1)           
    
    actor_scheduler = get_cosine_schedule_with_warmup(actor_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    # ================= 6. checkpoint 检查 =================
    current_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.run_name}_{arch_signature}")
    checkpoint_path = f"{current_ckpt_dir}/grpo_checkpoint.pth"
    
    # load_checkpoint 内部需要移除 critic 的加载
    load_epoch, load_step, swanlab_id = load_checkpoint(
        actor, actor_optimizer, actor_scheduler, scaler, checkpoint_path, device, is_distributed, strict=False
    )

    if load_epoch != -1:
        is_loaded = True
        start_epoch, start_step = load_epoch, load_step
    else:
        is_loaded = False
        start_epoch = start_step = 0

    if is_main_process:
        print(f"使用设备：{device}，是否开启分布式: {is_distributed}")
        if is_loaded:
            print(f"已从 checkpoint 中恢复训练，从第 {start_epoch + 1} 个 epoch 和第 {start_step + 1} 个 step 开始训练...")
        else:
            print("未找到 checkpoint，开始从零训练...") 
        print(f"{'-'*50}")

    if is_main_process and args.use_swanlab:
        import swanlab 
        swanlab_id = swanlab_id if swanlab_id else None
        run = swanlab.init(project=args.project_name, name=args.run_name, config=vars(args), id=swanlab_id, resume=swanlab_id)
        swanlab_id = getattr(swanlab, 'id', None)

    # ================= 7. 训练循环 =================
    actor.train()
    actor_optimizer.zero_grad()
    
    global_update_step = 0
    if is_loaded:
        epoch_update_steps = load_epoch * full_dataloader_len * args.grpo_epochs
        global_update_step = epoch_update_steps + (load_step + 1) * args.grpo_epochs    
        
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if epoch == start_epoch and is_loaded:
            steps_to_skip = start_step + 1 
            sampler = SkipStepSampler(base_sampler, steps_to_skip, args.rollout_batch_size)
        else:
            sampler = base_sampler

        dataloader = DataLoader(
            dataset, batch_size=args.rollout_batch_size, sampler=sampler, shuffle=False,      
            num_workers=0, pin_memory=True, drop_last=True 
        )

        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        for step, prompt_batch in enumerate(dataloader):
            if epoch == start_epoch and is_loaded:
                real_step = step + start_step + 1
            else:
                real_step = step
            
            # --- 阶段 A：经验收集 (Rollout) ---
            with torch.no_grad():
                # generate_grpo_experience 内部需要为每个 prompt 生成 G 个回复
                # 返回的数据维度通常会变成 (BatchSize * G, SeqLen)
                experience = generate_grpo_experience(
                    actor, ref_model, reward_model, 
                    prompt_batch, tokenizer, device, ptdtype,
                    group_size=args.group_size, # 传入组大小 G
                    max_response_length=args.max_response_length
                )
                
                # 计算组内相对 Advantage
                # 不需要 GAE，无需 value，直接算 rewards 的组内归一化
                # KL 散度惩罚通常在 compute_grpo_advantages 中从 reward 中扣除，或者在此处与 logprobs 结合
                advantages = compute_grpo_advantages(
                    experience['rewards'], 
                    experience['kl_penalties'], # 也可以在这里扣除 ref_model 和 actor 的 KL
                    experience['response_mask'],
                    group_size=args.group_size
                )

            # --- 阶段 B：GRPO 模型更新 ---
            running_actor_loss = 0.0
            
            for grpo_epoch in range(args.grpo_epochs):
                # 注意：此时的总数据量是 rollout_batch_size * group_size
                total_size = len(experience['prompts'])
                num_mini_batches = args.accumulation_steps
                prompt_len = experience['prompts'].shape[1]

                for i in range(num_mini_batches):
                    start_idx = i * total_size // num_mini_batches
                    end_idx = (i + 1) * total_size // num_mini_batches
                    
                    mb_inputs = experience['input_ids'][start_idx:end_idx]
                    mb_old_logprobs = experience['logprobs'][start_idx:end_idx]
                    mb_advantages = advantages[start_idx:end_idx]
                    mb_mask = experience['response_mask'][start_idx:end_idx]
                    mb_attention_mask = experience['attention_mask'][start_idx:end_idx]
                    
                    is_update_step = (i + 1) == num_mini_batches
                    actor_sync = actor.no_sync if is_distributed and not is_update_step else nullcontext

                    with actor_sync():
                        with torch.autocast(device_type="cuda", dtype=ptdtype):
                            
                            # ========== 计算 Actor (Policy) Loss ==========
                            actor_outputs = actor(mb_inputs, attention_mask=mb_attention_mask)
                            actor_logits = actor_outputs.logits[:, prompt_len - 1 : -1, :]
                            
                            new_logprobs = gather_logprobs(actor_logits, experience['actions'][start_idx:end_idx])
                            new_logprobs = new_logprobs * mb_mask
                            
                            # 重要性采样比率
                            ratio = torch.exp(new_logprobs - mb_old_logprobs)
                            
                            # Policy Surrogate Objective 保持与 PPO 一致
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            
                            # GRPO 也可以直接在 Loss 中显式加上 KL Penalty，具体取决于你的底层实现
                            # 如果你在 compute_grpo_advantages 中已经处理了 KL 奖励，这里就可以保持原样
                            actor_loss_unreduced = torch.max(pg_loss1, pg_loss2) * mb_mask
                            actor_loss = actor_loss_unreduced.sum() / (mb_mask.sum() + 1e-8)

                            actor_loss_scaled = actor_loss / num_mini_batches
                            running_actor_loss += actor_loss_scaled.detach().float().item()

                        # 仅 Actor 反向传播
                        scaler.scale(actor_loss_scaled).backward()

                    if is_update_step:
                        scaler.unscale_(actor_optimizer)
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        
                        actor_scale_before = scaler.get_scale()
                        scaler.step(actor_optimizer)
                        scaler.update()
                        actor_scale_after = scaler.get_scale()
                        
                        if actor_scale_before <= actor_scale_after:
                            actor_scheduler.step()
                        
                        actor_optimizer.zero_grad()
                        global_update_step += 1

                        # --- 日志记录 ---
                        if global_update_step % args.log_interval == 0 and is_main_process:
                            log_training_progress(
                                start_epoch, start_step, step=real_step, epoch=epoch, epochs=args.epochs,
                                dataloader_len=full_dataloader_len,
                                total_steps=args.epochs * full_dataloader_len,
                                loss_val=running_actor_loss, # 只有 Actor Loss 了
                                aux_loss_val=0.0, # Critic Loss 为 0
                                lr=actor_optimizer.param_groups[0]['lr'],
                                start_time=start_time,
                                use_swanlab=args.use_swanlab,
                            )
                        
                        running_actor_loss = 0.0

                        # --- 保存 Checkpoint ---
                        if global_update_step > 0 and global_update_step % args.save_steps == 0 and is_main_process:
                            save_checkpoint(
                                actor, actor_optimizer, actor_scheduler, scaler, epoch, step, checkpoint_path, 
                                is_distributed, swanlab_id
                            )
                            
                del actor_outputs, actor_loss, new_logprobs
                torch.cuda.empty_cache()

        # 每个 Epoch 结束后保存权重
        if is_main_process:
            actor_state = actor.module.state_dict() if is_distributed else actor.state_dict()
            
            output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
            os.makedirs(output_dir, exist_ok=True)
            
            tag = "final" if epoch == (args.epochs - 1) else f"epoch_{epoch}"
            torch.save(actor_state, f"{output_dir}/grpo_actor_{tag}.pth")
            print(f"Epoch {epoch} 结束，Actor 权重已保存！")

    total_time = time.time() - start_time
    print(f"GRPO 训练完成！总耗时: {datetime.timedelta(seconds=int(total_time))}")
    
    if is_main_process and args.use_swanlab:
        swanlab.finish()

    if is_distributed:
        dist.destroy_process_group()