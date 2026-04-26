"""
SFT (LoRA) 微调入口（Web 后端调用版）
将 train_sft_lora.py 的主逻辑封装为可调用的 run_sft(config_dict) 函数。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
import math
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from model.configuration import TinyuConfig
from dataset.lm_dataset import SFTDataset
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import build_sft_args_from_dict
from trainer.train_utils import init_model, set_seed, save_checkpoint, load_checkpoint, SkipStepSampler
from trainer.train_utils import setup_device_and_distributed, log_training_progress
from trainer.lora_utils import inject_custom_lora, print_trainable_parameters
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def run_sft(config: dict):
    """
    接收一个配置字典，执行 SFT LoRA 微调。
    """
    stop_event = config.pop("_stop_event", None)
    metrics_collector = config.pop("_metrics_collector", None)

    # ================= 0. 获取全局配置 =================
    args = build_sft_args_from_dict(config)
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_moe{int(args.use_moe)}"

    # ================= 1. 初始化分布式环境 =================
    device, local_rank, is_distributed, is_main_process = setup_device_and_distributed()
    set_seed(args.seed + local_rank)

    # ================= 2. 初始化模型并加载预训练权重 =================
    config_model = TinyuConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )

    model, tokenizer = init_model(config_model, tokenizer_path='./model', device=device)

    pretrained_dir = os.path.join(args.output_dir, f"{args.pretrain_run_name}_{arch_signature}")
    if is_main_process:
        os.makedirs(pretrained_dir, exist_ok=True)
    fallback_path = os.path.join(pretrained_dir, "pretrain_weight.pth")

    pretrained_path_to_load = None

    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        pretrained_path_to_load = args.pretrained_model_path
        if is_main_process:
            print(f" 找到指定路径的预训练权重: {pretrained_path_to_load}")
    elif os.path.exists(fallback_path):
        pretrained_path_to_load = fallback_path
        if is_main_process:
            print(f" 未找到指定权重，使用备用路径的预训练权重: {pretrained_path_to_load}")
    else:
        if is_main_process:
            print(f"警告: 指定路径 [{args.pretrained_model_path}] 和备用路径 [{fallback_path}] 均未找到权重文件！")
            print(f"正在对【随机初始化】的 {arch_signature} 模型进行 SFT (仅供 Debug)！")

    if pretrained_path_to_load:
        state_dict = torch.load(pretrained_path_to_load, map_location=device)
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            clean_state_dict[new_key] = v
        model.load_state_dict(clean_state_dict)
        if is_main_process:
            print("预训练基座权重加载成功！即将注入 LoRA...")

    # 注入 LoRA
    model = inject_custom_lora(
        model=model,
        target_modules=args.target_modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    if is_main_process:
        print_trainable_parameters(model)

    # ================= 3. DDP 包装 =================
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ================= 4. 数据集与优化器 =================
    dataset = SFTDataset(args.data_path, tokenizer, args.max_length)

    if is_distributed:
        base_sampler = DistributedSampler(dataset)
    else:
        from torch.utils.data import RandomSampler
        base_sampler = RandomSampler(dataset)

    base_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=base_sampler, drop_last=True
    )
    full_dataloader_len = len(base_dataloader)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # ================= 5. 初始化混合精度与余弦退火 =================
    ptdtype = torch.float16
    if args.dtype == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            ptdtype = torch.bfloat16
            if is_main_process:
                print("硬件支持 BF16，已开启 Bfloat16 混合精度训练")
        else:
            if is_main_process:
                print("当前硬件不支持 BF16，已自动降级为 Float16")
            ptdtype = torch.float16
    elif args.dtype == "float32":
        ptdtype = torch.float32

    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

    total_update_steps = args.epochs * math.ceil(full_dataloader_len / args.accumulation_steps)
    warmup_steps = int(total_update_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps
    )

    # ================= 6. checkpoint 检查 =================
    current_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.run_name}_{arch_signature}")
    checkpoint_path = f"{current_ckpt_dir}/sft_checkpoint.pth"
    load_epoch, load_step, swanlab_id = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_path, device, is_distributed, strict=False
    )

    if load_epoch != -1:
        is_loaded = True
        start_epoch, start_step = load_epoch, load_step
    else:
        is_loaded = False
        start_epoch = start_step = 0

    if is_main_process:
        print(f"使用设备：{device}，是否开启分布式: {is_distributed}")
        print(f"已固定全局随机种子为: {args.seed}")
        if is_loaded:
            print(f"已从 checkpoint 中恢复训练，从第 {start_epoch + 1} 个 epoch 和第 {start_step + 1} 个 step 开始训练...")
        else:
            print("未找到 checkpoint，开始从零训练...")
        print(f"{'-' * 50}")

    if is_main_process and args.use_swanlab:
        import swanlab
        swanlab_id = swanlab_id if swanlab_id else None
        run = swanlab.init(
            project=args.project_name,
            name=args.run_name,
            config=vars(args),
            id=swanlab_id,
            resume=swanlab_id
        )
        if hasattr(swanlab, 'get_run'):
            run = swanlab.get_run()
            swanlab_id = getattr(run, 'id', None) if run else None
        else:
            swanlab_id = getattr(swanlab, 'id', None)

    # ================= 5. SFT 训练循环 =================
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    running_aux_loss = 0.0
    global_update_step = 0

    if is_loaded:
        epoch_update_steps = math.ceil((load_epoch * full_dataloader_len) / args.accumulation_steps)
        global_update_step = epoch_update_steps + math.ceil((load_step + 1) / args.accumulation_steps)

    if is_main_process:
        print(f"已更新参数次数 {global_update_step}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # --- 外部停止信号检查 ---
        if stop_event and stop_event.is_set():
            if is_main_process:
                print("收到停止信号，正在结束训练...")
            break

        if epoch == start_epoch and is_loaded:
            steps_to_skip = start_step + 1
            if is_main_process:
                print(f"正在从 Sampler 层面直接跳过前 {steps_to_skip} 个 Step 的数据...")
            sampler = SkipStepSampler(base_sampler, steps_to_skip, args.batch_size)
        else:
            if is_main_process:
                print(f"Epoch {epoch}: 使用完整数据集训练...")
            sampler = base_sampler

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        for step, (input_ids, labels) in enumerate(dataloader):
            if stop_event and stop_event.is_set():
                if is_main_process:
                    print("收到停止信号，正在结束训练...")
                break

            if epoch == start_epoch and is_loaded:
                real_step = step + start_step + 1
            else:
                real_step = step

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            is_update_step = (real_step + 1) % args.accumulation_steps == 0 or (real_step + 1) == full_dataloader_len

            sync_context = model.no_sync if is_distributed and not is_update_step else nullcontext
            with sync_context():
                with torch.autocast(device_type="cuda", dtype=ptdtype):
                    outputs = model(input_ids, labels=labels)

                    raw_main_loss = outputs.loss
                    raw_aux_loss = outputs.aux_loss
                    loss = raw_main_loss + raw_aux_loss

                    if (real_step + 1) == full_dataloader_len:
                        current_accum_steps = full_dataloader_len % args.accumulation_steps
                        if current_accum_steps == 0:
                            current_accum_steps = args.accumulation_steps
                    else:
                        current_accum_steps = args.accumulation_steps

                    running_loss += raw_main_loss.detach().float()
                    running_aux_loss += raw_aux_loss.detach().float()

                    loss = loss / current_accum_steps

                scaler.scale(loss).backward()

            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                optimizer.zero_grad()

                global_update_step += 1

                if global_update_step % args.log_interval == 0 and is_main_process:
                    avg_loss = (running_loss / current_accum_steps).item()
                    avg_aux_loss = (running_aux_loss / current_accum_steps).item()

                    log_training_progress(
                        start_epoch,
                        start_step,
                        step=real_step,
                        epoch=epoch,
                        epochs=args.epochs,
                        dataloader_len=full_dataloader_len,
                        total_steps=args.epochs * full_dataloader_len,
                        loss_val=avg_loss,
                        aux_loss_val=avg_aux_loss,
                        lr=optimizer.param_groups[0]['lr'],
                        start_time=start_time,
                        use_swanlab=args.use_swanlab,
                    )

                    if metrics_collector:
                        metrics_collector.record(
                            step=global_update_step,
                            loss=avg_loss,
                            aux_loss=avg_aux_loss,
                            lr=optimizer.param_groups[0]['lr']
                        )

                running_loss = 0.0
                running_aux_loss = 0.0

                if global_update_step > 0 and global_update_step % args.save_steps == 0 and is_main_process:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, real_step,
                        checkpoint_path, is_distributed, swanlab_id,
                        only_lora=True
                    )

        if stop_event and stop_event.is_set():
            break

        # ================= 6. 保存 LoRA 权重 =================
        if is_main_process:
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
            os.makedirs(output_dir, exist_ok=True)
            lora_state = {k: v for k, v in state_dict.items() if 'lora_A' in k or 'lora_B' in k}
            if epoch == (args.epochs - 1):
                torch.save(lora_state, f"{output_dir}/lora_weight.pth")
            else:
                torch.save(lora_state, f"{output_dir}/lora_epoch_{epoch}.pth")
            print(f"Epoch {epoch} 完成，LoRA 权重已保存至 {output_dir}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"训练完成！总耗时: {total_time_str}")

    if is_main_process and args.use_swanlab:
        swanlab.finish()

    if is_distributed:
        dist.destroy_process_group()
