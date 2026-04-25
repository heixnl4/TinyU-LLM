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
from dataset.lm_dataset import PretrainDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_pretrain_args
from trainer.train_utils import SkipStepSampler, print_model_param_details, init_model, set_seed, save_checkpoint
from trainer.train_utils import load_checkpoint, setup_device_and_distributed, log_training_progress
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext


if __name__ == "__main__":
    # ================= 0. 获取全局配置 =================
    args = parse_pretrain_args()
    # 用模型的核心架构参数生成一个“架构签名”
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_moe{int(args.use_moe)}"

    # ================= 1. 初始化分布式环境 =================
    device, local_rank, is_distributed, is_main_process = setup_device_and_distributed()

    # 锁定全局随机种子 每张卡的种子稍有不同,防止 Dropout 等随机操作在多卡上完全一致
    set_seed(args.seed + local_rank)

    # ================= 2. 初始化模型与数据 =================
    config = TinyuConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )

    model, tokenizer = init_model(config, device=device)

    # 打印模型总参数量
    if is_main_process:
        print_model_param_details(model, detail=False)

    # =================3. 模型编译与 DDP 包装 =================
    # 模型编译 (必须在 DDP 之前进行)
    if args.use_compile:
        if is_main_process: print("正在使用 torch.compile 编译模型...")
        model = torch.compile(model)

    # DDP 包装
    if is_distributed:
        # 告诉 DDP 忽略掉 RoPE 的缓存参数，否则会报错
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DDP(model, device_ids=[local_rank])

    # ================= 4. 数据集与优化器 =================
    dataset = PretrainDataset(args.data_path, tokenizer, args.max_length)

    if is_distributed:
        # 分布式采样器：确保不同的 GPU 拿到不同的数据切片，不会重复训练
        base_sampler = DistributedSampler(dataset)
        # 用 sampler 时 shuffle 必须为 False
    else:
        from torch.utils.data import RandomSampler, SequentialSampler
        base_sampler = RandomSampler(dataset)

    # 先创建一个基础的、不跳步的 DataLoader，仅用于计算正确的总步数
    base_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=base_sampler, drop_last=True
    )
    full_dataloader_len = len(base_dataloader)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # ================= 5. 初始化混合精度与余弦退火 =================
    # 智能判断并选择混合精度的 dtype
    ptdtype = torch.float16
    if args.dtype == "bfloat16":
        # 检查显卡是否真的支持 BF16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            ptdtype = torch.bfloat16
            if is_main_process: print("硬件支持 BF16，已开启 Bfloat16 混合精度训练")
        else:
            if is_main_process: print("当前硬件不支持 BF16，已自动降级为 Float16")
            ptdtype = torch.float16
    elif args.dtype == "float32":
        ptdtype = torch.float32

    # BF16 的动态范围足够大，不需要缩放梯度；如果设为 False，scaler 所有的操作都会变成空操作（无消耗）
    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

    # 计算真实的总更新步数
    # 确保与实际发生的 update_step 次数绝对一致
    total_update_steps = args.epochs * math.ceil(full_dataloader_len / args.accumulation_steps)    
    # 拿出前 10% 的步数做预热 (Warmup)
    warmup_steps = int(total_update_steps * 0.1)           
    # 初始化学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_update_steps
    )

    # ================= 6. checkpoint 检查 =================
    
    # 按 run_name 和 架构签名 来建立独立的 Checkpoint 文件夹
    # 比如：./checkpoints/simple-test-run-01_h256_l2_ah4_moe1
    current_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.run_name}_{arch_signature}")

    # 尝试加载 Checkpoint
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
        print(f"{'-'*50}")
        

    if is_main_process and args.use_swanlab:
        import swanlab 
        # 初始化 SwanLab：创建一个实验，并记录下这次实验的"超参数"
        swanlab_id = swanlab_id if swanlab_id else None
        resume = 'must' if swanlab_id else None
        run = swanlab.init(
            project=args.project_name,  # 项目名称（相当于一个大文件夹）
            name=args.run_name, # 这次实验的名称
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
    # 清空梯度的操作必须移到循环外面（或者更新权重的后面）
    optimizer.zero_grad()
    # 初始化用于累积真实 Loss 的变量
    running_loss = 0.0
    running_aux_loss = 0.0
    # 记录全局更新步数
    global_update_step = 0
    if is_loaded:
        # 根据已练过的数据量推算真实的全局更新步数
        epoch_update_steps = math.ceil((load_epoch * full_dataloader_len) / args.accumulation_steps) 
        global_update_step = epoch_update_steps + math.ceil((load_step + 1) / args.accumulation_steps)
    # 记录整个训练开始的时间
    start_time = time.time()

    # 如果是恢复训练则从 start_epoch 开始，如果没有 checkpoint 则 start_epoch 默认为0
    for epoch in range(start_epoch, args.epochs):
        # start_step 是上一次断点前最后完成的 step 索引
        # 判断是否需要跳过数据 (仅在恢复训练的那个 Epoch 生效)
        if epoch == start_epoch and is_loaded:
            steps_to_skip = start_step + 1 
            if is_main_process:
                print(f"正在从 Sampler 层面直接跳过前 {steps_to_skip} 个 Step 的数据...")
            sampler = SkipStepSampler(base_sampler, steps_to_skip, args.batch_size)
        else:
            if is_main_process:
                print(f"Epoch {epoch}: 使用完整数据集训练...")
            sampler = base_sampler

        # 3. 初始化 DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=sampler, 
            shuffle=False,      # 使用了 Sampler，这里必须是 False
            num_workers=0,      # 记得加上多进程
            pin_memory=True,    # 记得开启锁页内存
            drop_last=True
        )
        # 必须加上这句：让采样器打乱每个 epoch 的数据顺序
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        for step, (input_ids, labels) in enumerate(dataloader):
            # 因为我们用了 SkipSampler，此时 enumerate 的 step 是从 0 重新开始计数的。
            # 为了让日志和保存逻辑依然对齐真实的全局进度，我们需要把真实的 step 算出来：
            if epoch == start_epoch and is_loaded:
                real_step = step + start_step + 1
            else:
                real_step = step

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # 判断当前 step 是否是累积的最后一步
            is_update_step = (real_step + 1) % args.accumulation_steps == 0 or (real_step + 1) == full_dataloader_len
            
            # DDP 高级优化 防止无意义的梯度同步
            sync_context = model.no_sync if is_distributed and not is_update_step else nullcontext

            with sync_context():
                # 前向传播，开启自动混合精度上下文，with 块里的代码会以 fp16 运行
                with torch.autocast(device_type="cuda", dtype=ptdtype):
                    outputs = model(input_ids, labels=labels)

                    # 取出原始的 loss 值
                    raw_main_loss = outputs.loss
                    raw_aux_loss = outputs.aux_loss if outputs.aux_loss is not None else 0
                    loss = raw_main_loss + raw_aux_loss

                    # 动态计算当前的真实累积步数
                    if (real_step + 1) == full_dataloader_len:
                        # 算一下最后这一波攒了多少个 step
                        current_accum_steps = full_dataloader_len % args.accumulation_steps
                        if current_accum_steps == 0:
                            current_accum_steps = args.accumulation_steps
                    else:
                        current_accum_steps = args.accumulation_steps
                    
                    # loss 真实数值累加起来
                    # 2. 在 with sync_context() 内部修改累加逻辑：
                    # 核心：使用 .detach().float() 代替 .item()
                    # .detach() 将 tensor 从计算图中剥离，防止显存泄漏（OOM）
                    # .float() 转换精度，防止 fp16 累加溢出
                    running_loss += raw_main_loss.detach().float()
                    if outputs.aux_loss is not None:
                        running_aux_loss += raw_aux_loss.detach().float()

                    # 因为 backward() 默认是把梯度加起来，除以步数求平均值，等效于真实的大 Batch
                    loss = loss / current_accum_steps

                if real_step % args.log_interval == 0 and is_main_process:
                    print(f"反向传播前真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                # 无论是否更新权重，每一步都要反向传播累积梯度
                scaler.scale(loss).backward()

                if real_step % args.log_interval == 0 and is_main_process:
                    print(f"反向传播后真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    print(f"缓存池预留: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            if is_update_step:
                # 梯度裁剪：必须在 unscale 之后进行
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 记录更新前的 scale
                scale_before = scaler.get_scale()
                # 更新权重和 Scaler
                scaler.step(optimizer)
                scaler.update()
                # 记录更新后的 scale
                scale_after = scaler.get_scale()
                # 只有当 scale 没有减小（即没有发生 inf/nan 跳过更新）时，才更新学习率
                if scale_before <= scale_after:
                    # 更新学习率 (每个 step 走一步)
                    scheduler.step()
                
                # 更新完权重后，立刻清空积攒的梯度，迎接下一轮累积
                optimizer.zero_grad()

                # 基于全局更新步数来保存和上传日志
                global_update_step += 1

                # 3. 记录与上传日志
                if global_update_step % args.log_interval == 0 and is_main_process:
                    # 计算这一个大 Batch（Macro-batch）的真实平均 Loss
                    avg_loss = (running_loss / current_accum_steps).item()
                    # aux_loss 可能是浮点数 0，也可能是 Tensor
                    if isinstance(running_aux_loss, torch.Tensor):
                        avg_aux_loss = (running_aux_loss / current_accum_steps).item()
                    else:
                        avg_aux_loss = 0.0

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
                # 打印完并且更新完权重后，清空累积容器
                running_loss = 0.0
                running_aux_loss = 0.0
                
                # 保存一次 Checkpoint 
                if global_update_step > 0 and global_update_step % args.save_steps == 0 and is_main_process:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, real_step, 
                        checkpoint_path, is_distributed, swanlab_id
                    )

        # 每轮跑完存一个模型
        if is_main_process:
            # 如果被 DDP 包装过，需要用 model.module.state_dict() 取出真实权重
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

    # 销毁分布式进程组
    if is_distributed:
        dist.destroy_process_group()