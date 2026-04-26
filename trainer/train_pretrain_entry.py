"""
预训练入口（Web 后端调用版）
将 train_pretrain.py 的主逻辑封装为可调用的 run_pretrain(config_dict) 函数。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
import torch
import torch.distributed as dist
import math
from torch import optim
from torch.utils.data import DataLoader
from model.configuration import TinyuConfig
from dataset.lm_dataset import PretrainDataset
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import build_pretrain_args_from_dict
from trainer.train_utils import SkipStepSampler, print_model_param_details, init_model, set_seed, save_checkpoint
from trainer.train_utils import load_checkpoint, setup_device_and_distributed, log_training_progress
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext


def run_pretrain(config: dict):
    """
    接收一个配置字典，执行预训练。
    config 中可包含:
      - 所有训练超参（与命令行参数同名）
      - _stop_event: threading.Event，用于外部请求停止训练
      - _metrics_collector: MetricsCollector，用于向前端推送指标
    """
    stop_event = config.pop("_stop_event", None)
    metrics_collector = config.pop("_metrics_collector", None)

    # ================= 0. 获取全局配置 =================
    args = build_pretrain_args_from_dict(config)
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_moe{int(args.use_moe)}"

    # ================= 1. 初始化分布式环境 =================
    device, local_rank, is_distributed, is_main_process = setup_device_and_distributed()
    set_seed(args.seed + local_rank)

    # ================= 2. 初始化模型与数据 =================
    config_model = TinyuConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )

    model, tokenizer = init_model(config_model, tokenizer_path='./model', device=device)

    if is_main_process:
        print_model_param_details(model, detail=False)

    # ================= 3. 模型编译与 DDP 包装 =================
    if args.use_compile:
        if is_main_process:
            print("正在使用 torch.compile 编译模型...")
        model = torch.compile(model)

    if is_distributed:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DDP(model, device_ids=[local_rank])

    # ================= 4. 数据集与优化器 =================
    dataset = PretrainDataset(args.data_path, tokenizer, args.max_length)

    if is_distributed:
        base_sampler = DistributedSampler(dataset)
    else:
        from torch.utils.data import RandomSampler
        base_sampler = RandomSampler(dataset)

    base_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=base_sampler, drop_last=True
    )
    full_dataloader_len = len(base_dataloader)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

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
    checkpoint_path = f"{current_ckpt_dir}/pretrain_checkpoint.pth"
    load_epoch, load_step, swanlab_id = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_path, device, is_distributed
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

    # ================= 7. 训练循环 =================
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    running_aux_loss = 0.0
    global_update_step = 0
    if is_loaded:
        epoch_update_steps = math.ceil((load_epoch * full_dataloader_len) / args.accumulation_steps)
        global_update_step = epoch_update_steps + math.ceil((load_step + 1) / args.accumulation_steps)
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
            # --- 每步检查停止信号 ---
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
                    raw_aux_loss = outputs.aux_loss if outputs.aux_loss is not None else 0
                    loss = raw_main_loss + raw_aux_loss

                    if (real_step + 1) == full_dataloader_len:
                        current_accum_steps = full_dataloader_len % args.accumulation_steps
                        if current_accum_steps == 0:
                            current_accum_steps = args.accumulation_steps
                    else:
                        current_accum_steps = args.accumulation_steps

                    running_loss += raw_main_loss.detach().float()
                    if outputs.aux_loss is not None:
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
                    if isinstance(running_aux_loss, torch.Tensor):
                        avg_aux_loss = (running_aux_loss / current_accum_steps).item()
                    else:
                        avg_aux_loss = 0.0

                    # --- 打印日志 ---
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

                    # --- 推送指标到前端 ---
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
                        checkpoint_path, is_distributed, swanlab_id
                    )

        # 检查是否因停止信号跳出内层循环
        if stop_event and stop_event.is_set():
            break

        # 每轮跑完存一个模型
        if is_main_process:
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
            os.makedirs(output_dir, exist_ok=True)
            if epoch == (args.epochs - 1):
                torch.save(state_dict, f"{output_dir}/pretrain_weight.pth")
            else:
                torch.save(state_dict, f"{output_dir}/pretrain_epoch_{epoch}.pth")
            print(f"Epoch {epoch} 完成并保存！")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"训练完成！总耗时: {total_time_str}")

    if is_main_process and args.use_swanlab:
        swanlab.finish()

    if is_distributed:
        dist.destroy_process_group()
