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
from dataset.lm_dataset import DPODataset
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_dpo_args
from trainer.dpo_utils import init_dpo_models, get_batch_logprobs, compute_dpo_loss
from trainer.train_utils import print_model_param_details, set_seed, save_checkpoint, SkipStepSampler
from trainer.train_utils import load_checkpoint, setup_device_and_distributed, log_training_progress
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

if __name__ == "__main__":
    # ================= 0. 获取全局配置 =================
    args = parse_dpo_args()
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_dpo"

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

    actor, ref_model, tokenizer = init_dpo_models(config, args.actor_model_path, device=device)

    # 冻结 Reference 模型
    ref_model.eval()
    for param in ref_model.parameters():
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
    dataset = DPODataset(args.data_path, tokenizer, args.max_prompt_length, args.max_response_length)

    if is_distributed:
        base_sampler = DistributedSampler(dataset)
    else:
        from torch.utils.data import RandomSampler
        base_sampler = RandomSampler(dataset)

    base_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=base_sampler, drop_last=True
    )
    full_dataloader_len = len(base_dataloader)

    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.learning_rate)

    # ================= 5. 初始化混合精度与余弦退火 =================
    ptdtype = torch.float16
    if args.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
        if is_main_process: print("硬件支持 BF16，已开启 Bfloat16 混合精度训练")
    elif args.dtype == "float32":
        ptdtype = torch.float32

    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

    total_update_steps = (args.epochs * full_dataloader_len) // args.accumulation_steps
    warmup_steps = int(total_update_steps * 0.1)
    actor_scheduler = get_cosine_schedule_with_warmup(actor_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    # ================= 6. checkpoint 检查 =================
    current_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.run_name}_{arch_signature}")
    checkpoint_path = f"{current_ckpt_dir}/dpo_checkpoint.pth"

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

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if epoch == start_epoch and is_loaded:
            steps_to_skip = start_step + 1
            sampler = SkipStepSampler(base_sampler, steps_to_skip, args.batch_size)
        else:
            sampler = base_sampler

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=True
        )

        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        running_loss, running_acc = 0.0, 0.0

        for step, batch in enumerate(dataloader):
            if epoch == start_epoch and is_loaded:
                real_step = step + start_step + 1
            else:
                real_step = step

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            prompt_lens = batch["prompt_length"].to(device)

            is_update_step = ((step + 1) % args.accumulation_steps == 0) or ((step + 1) == len(dataloader))
            actor_sync = actor.no_sync if is_distributed and not is_update_step else nullcontext

            with actor_sync():
                with torch.autocast(device_type="cuda", dtype=ptdtype):
                    concatenated_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
                    concatenated_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
                    concat_prompt_lens = torch.cat([prompt_lens, prompt_lens], dim=0)

                    policy_logits = actor(concatenated_ids, attention_mask=concatenated_mask).logits
                    all_policy_logprobs = get_batch_logprobs(policy_logits, concatenated_ids, concatenated_mask, concat_prompt_lens.unsqueeze(-1))
                    policy_chosen_logprobs, policy_rejected_logprobs = all_policy_logprobs.chunk(2)

                    with torch.no_grad():
                        ref_logits = ref_model(concatenated_ids, attention_mask=concatenated_mask).logits
                        all_ref_logprobs = get_batch_logprobs(ref_logits, concatenated_ids, concatenated_mask, concat_prompt_lens.unsqueeze(-1))
                        ref_chosen_logprobs, ref_rejected_logprobs = all_ref_logprobs.chunk(2)

                    loss, reward_acc = compute_dpo_loss(
                        policy_chosen_logprobs, policy_rejected_logprobs,
                        ref_chosen_logprobs, ref_rejected_logprobs,
                        beta=args.beta
                    )

                    loss_scaled = loss / args.accumulation_steps

                scaler.scale(loss_scaled).backward()

                running_loss += loss.detach().float().item()
                running_acc += reward_acc.detach().float().item()

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
                    avg_loss = running_loss / args.accumulation_steps
                    avg_acc = running_acc / args.accumulation_steps
                    log_training_progress(
                        start_epoch, start_step, step=real_step, epoch=epoch, epochs=args.epochs,
                        dataloader_len=full_dataloader_len,
                        total_steps=args.epochs * full_dataloader_len,
                        loss_val=avg_loss,
                        aux_loss_val=0.0,
                        lr=actor_optimizer.param_groups[0]['lr'],
                        start_time=start_time,
                        use_swanlab=args.use_swanlab,
                        reward_acc=avg_acc,
                    )
                    running_loss, running_acc = 0.0, 0.0

                # --- 保存 Checkpoint ---
                if global_update_step > 0 and global_update_step % args.save_steps == 0 and is_main_process:
                    save_checkpoint(
                        actor, actor_optimizer, actor_scheduler, scaler, epoch, step, checkpoint_path,
                        is_distributed, swanlab_id
                    )

        # 每个 Epoch 结束后保存权重
        if is_main_process:
            actor_state = actor.module.state_dict() if is_distributed else actor.state_dict()

            output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
            os.makedirs(output_dir, exist_ok=True)

            tag = "final" if epoch == (args.epochs - 1) else f"epoch_{epoch}"
            torch.save(actor_state, f"{output_dir}/dpo_actor_{tag}.pth")
            print(f"Epoch {epoch} 结束，Actor 权重已保存！")

    total_time = time.time() - start_time
    print(f"DPO 训练完成！总耗时: {datetime.timedelta(seconds=int(total_time))}")

    if is_main_process and args.use_swanlab:
        swanlab.finish()

    if is_distributed:
        dist.destroy_process_group()
