import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
import torch
import math
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from model.configuration import TinyuConfig
# PPO 需要专门的 Prompt 数据集
from dataset.lm_dataset import PromptDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_ppo_args
# 假设你在 utils 中补充了 PPO 相关的初始化和 GAE 计算函数
from trainer.ppo_utils import init_ppo_models, generate_experience, compute_gae, gather_logprobs
from trainer.train_utils import print_model_param_details, set_seed, save_checkpoint
from trainer.train_utils import load_checkpoint, setup_device_and_distributed, log_training_progress
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext
import torch.nn.functional as F

if __name__ == "__main__":
    # ================= 0. 获取全局配置 =================
    args = parse_ppo_args()
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_ppo"

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

    # PPO 需要 4 个模型: 
    # 1. actor: 正在训练的策略模型 (从 SFT 初始化)
    # 2. critic: 正在训练的价值模型 (从 Reward 初始化)
    # 3. ref_model: 参考模型，用于计算 KL 散度 (冻结，从 SFT 初始化)
    # 4. reward_model: 奖励模型，用于打分 (冻结)
    actor, critic, ref_model, reward_model, tokenizer = init_ppo_models(config, args.model_paths, device=device)

    # 冻结 Reference 和 Reward 模型
    for model in [ref_model, reward_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    if is_main_process:
        print("Actor Model Params:")
        print_model_param_details(actor, detail=False)
        print("Critic Model Params:")
        print_model_param_details(critic, detail=False)

    # ================= 3. 模型编译与 DDP 包装 =================
    if args.use_compile:
        if is_main_process: print("正在使用 torch.compile 编译模型...")
        actor = torch.compile(actor)
        critic = torch.compile(critic)

    if is_distributed:
        actor._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 只有 actor 和 critic 需要被 DDP 包装并进行梯度同步
        actor = DDP(actor, device_ids=[local_rank])
        critic = DDP(critic, device_ids=[local_rank])

    # ================= 4. 数据集与优化器 =================
    # PPO 阶段输入通常只有 Prompts，模型自己生成 Responses
    dataset = PromptDataset(args.data_path, tokenizer, args.max_prompt_length)

    if is_distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.rollout_batch_size, sampler=sampler, shuffle=False) 
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.rollout_batch_size, shuffle=True)

    # Actor 和 Critic 通常使用不同的学习率（Critic 往往需要更大的 LR）
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.critic_learning_rate)

    # ================= 5. 初始化混合精度与余弦退火 =================
    ptdtype = torch.float16
    if args.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
        if is_main_process: print("硬件支持 BF16，已开启 Bfloat16 混合精度训练")
    elif args.dtype == "float32":
        ptdtype = torch.float32

    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

    # PPO 的总更新步数 = (外层 Epoch * DataLoader长度) * (PPO 内部 Epoch 数量) / 累积步数
    total_rollouts = args.epochs * len(dataloader)
    total_update_steps = total_rollouts * args.ppo_epochs // args.accumulation_steps 
    warmup_steps = int(total_update_steps * 0.1)           
    
    actor_scheduler = get_cosine_schedule_with_warmup(actor_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
    critic_scheduler = get_cosine_schedule_with_warmup(critic_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    # ================= 6. checkpoint 检查 =================
    current_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.run_name}_{arch_signature}")
    checkpoint_path = f"{current_ckpt_dir}/ppo_checkpoint.pth"
    
    # 需自行在 load_checkpoint 中兼容双 Optimizer 和双 Scheduler 的加载逻辑
    start_epoch, start_step, swanlab_id = load_checkpoint(
        actor, actor_optimizer, actor_scheduler, scaler, checkpoint_path, device, is_distributed, 
        critic, critic_optimizer, critic_scheduler
    )

    if is_main_process and args.use_swanlab:
        import swanlab 
        swanlab_id = swanlab_id if swanlab_id else None
        run = swanlab.init(project=args.project_name, name=args.run_name, config=vars(args), id=swanlab_id, resume=swanlab_id)
        swanlab_id = getattr(swanlab, 'id', None)

    # ================= 7. 训练循环 =================
    actor.train()
    critic.train()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    global_update_step = 0
    if start_epoch > 0 or start_step > 0:
        global_update_step = (start_epoch * len(dataloader) + start_step + 1) // args.accumulation_steps
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None: 
            sampler.set_epoch(epoch)

        for step, prompt_batch in enumerate(dataloader):
            if epoch == start_epoch and step <= start_step:
                continue
            
            # --- 阶段 A：经验收集 (Rollout) ---
            # 此阶段不需要梯度，使用 Actor 生成文本，并利用 Ref 和 Reward 模型评估
            with torch.no_grad():
                # 1. 生成 Response，并计算 Old Logprobs, Values, Rewards, 和 KL Penalty
                # 这里 generate_experience 是一个高度封装的函数
                experience = generate_experience(
                    actor, critic, ref_model, reward_model, 
                    prompt_batch, tokenizer, device, ptdtype
                )
                
                # 2. 计算 GAE (广义优势估计) 和 Returns
                advantages, returns = compute_gae(
                    experience['rewards'], experience['values'], 
                    gamma=args.gamma, lam=args.gae_lambda
                )
                
                # 标准化优势函数，提升训练稳定性
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- 阶段 B：PPO 模型更新 ---
            # 对收集到的一批经验进行多次 PPO 迭代更新
            
            # 初始化 PPO epoch 的 loss 记录
            running_actor_loss, running_critic_loss = 0.0, 0.0
            
            for ppo_epoch in range(args.ppo_epochs):
                # 如果显存不够，可以将 experience 切分成更小的 mini_batch
                # 这里为了配合你原本的 accum 结构，假设我们直接在完整 batch 上进行累积切片
                
                # 模拟大 Batch 分步前向以节省显存 (Gradient Accumulation)
                num_mini_batches = args.accumulation_steps
                mini_batch_size = max(1, len(experience['prompts']) // num_mini_batches)

                for i in range(num_mini_batches):
                    # 获取 mini-batch 数据切片
                    start_idx = i * mini_batch_size
                    end_idx = start_idx + mini_batch_size
                    mb_inputs = experience['input_ids'][start_idx:end_idx]
                    mb_old_logprobs = experience['logprobs'][start_idx:end_idx]
                    mb_advantages = advantages[start_idx:end_idx]
                    mb_returns = returns[start_idx:end_idx]
                    mb_old_values = experience['values'][start_idx:end_idx]
                    
                    is_update_step = (i + 1) == num_mini_batches

                    # DDP 上下文同步
                    actor_sync = actor.no_sync if is_distributed and not is_update_step else nullcontext
                    critic_sync = critic.no_sync if is_distributed and not is_update_step else nullcontext

                    with actor_sync(), critic_sync():
                        with torch.autocast(device_type="cuda", dtype=ptdtype):
                            
                            # ========== 1. 计算 Actor (Policy) Loss ==========
                            # 重新计算当前策略的 logprob
                            actor_outputs = actor(mb_inputs)
                            # 假设 get_logprobs 是从 logits 中提取 action 对应 logprob 的函数
                            new_logprobs = gather_logprobs(actor_outputs.logits, experience['actions'][start_idx:end_idx])
                            
                            # 重要性采样比率 (Importance Sampling Ratio)
                            ratio = torch.exp(new_logprobs - mb_old_logprobs)
                            
                            # PPO 截断机制 (Clipped Surrogate Objective)
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # ========== 2. 计算 Critic (Value) Loss ==========
                            new_values = critic(mb_inputs).squeeze(-1)
                            # Value Clip 机制，防止 Value 网络更新跨度过大
                            vpredclipped = mb_old_values + torch.clamp(new_values - mb_old_values, -args.cliprange_value, args.cliprange_value)
                            vf_losses1 = (new_values - mb_returns) ** 2
                            vf_losses2 = (vpredclipped - mb_returns) ** 2
                            critic_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()

                            # 按累积步数平均 Loss
                            actor_loss_scaled = actor_loss / num_mini_batches
                            critic_loss_scaled = critic_loss / num_mini_batches

                            # 记录纯净的 Loss (脱离计算图)
                            running_actor_loss += actor_loss_scaled.detach().float()
                            running_critic_loss += critic_loss_scaled.detach().float()

                        # 分别反向传播
                        scaler.scale(actor_loss_scaled).backward()
                        scaler.scale(critic_loss_scaled).backward()

                    if is_update_step:
                        # 梯度裁剪
                        scaler.unscale_(actor_optimizer)
                        scaler.unscale_(critic_optimizer)
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                        
                        # Actor 更新
                        actor_scale_before = scaler.get_scale()
                        scaler.step(actor_optimizer)
                        # Critic 更新
                        scaler.step(critic_optimizer)
                        
                        scaler.update()
                        actor_scale_after = scaler.get_scale()
                        
                        if actor_scale_before <= actor_scale_after:
                            actor_scheduler.step()
                            critic_scheduler.step()
                        
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        
                        global_update_step += 1

                        # --- 日志记录 ---
                        if global_update_step % args.log_interval == 0 and is_main_process:
                            log_training_progress(
                                start_epoch, start_step, step=step, epoch=epoch, epochs=args.epochs,
                                loss_val=running_actor_loss.item(), # PPO 主 Loss
                                aux_loss_val=running_critic_loss.item(), # 将 Critic Loss 作为 Aux 记录
                                lr=actor_optimizer.param_groups[0]['lr'],
                                start_time=start_time,
                                use_swanlab=args.use_swanlab,
                            )
                        
                        running_actor_loss, running_critic_loss = 0.0, 0.0

                        # --- 保存 Checkpoint ---
                        if global_update_step > 0 and global_update_step % args.save_steps == 0 and is_main_process:
                            # 记得在保存逻辑里同时保存 actor 和 critic
                            save_checkpoint(
                                actor, actor_optimizer, actor_scheduler, scaler, epoch, step, 
                                checkpoint_path, is_distributed, swanlab_id
                            )

        # 每个 Epoch 结束后保存权重
        if is_main_process:
            actor_state = actor.module.state_dict() if is_distributed else actor.state_dict()
            critic_state = critic.module.state_dict() if is_distributed else critic.state_dict()
            
            output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
            os.makedirs(output_dir, exist_ok=True)
            
            tag = "final" if epoch == (args.epochs - 1) else f"epoch_{epoch}"
            torch.save(actor_state, f"{output_dir}/ppo_actor_{tag}.pth")
            torch.save(critic_state, f"{output_dir}/ppo_critic_{tag}.pth")
            print(f"Epoch {epoch} 结束，Actor & Critic 权重已保存！")

    total_time = time.time() - start_time
    print(f"PPO 训练完成！总耗时: {datetime.timedelta(seconds=int(total_time))}")
    
    if is_main_process and args.use_swanlab:
        swanlab.finish()

    if is_distributed:
        dist.destroy_process_group()