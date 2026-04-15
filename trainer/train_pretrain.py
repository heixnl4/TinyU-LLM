import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time           # 用于记录时间戳
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from model.configuration import TinyuConfig
from dataset.lm_dataset import PretrainDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_args        # 中央参数解析器
from trainer.train_utils import print_model_param_details, init_model,  set_seed, save_checkpoint
from trainer.train_utils import load_checkpoint, setup_device_and_distributed, log_training_progress
# 导入分布式计算需要的模块
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext



if __name__ == "__main__":
    # ================= 0. 获取全局配置 =================
    args = parse_args()

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

    # ================= 4. 数据集与 DistributedSampler =================
    dataset = PretrainDataset(args.data_path, tokenizer, args.max_length)

    if is_distributed:
        # 分布式采样器：确保不同的 GPU 拿到不同的数据切片，不会重复训练
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False) # 注意：用 sampler 时 shuffle 必须为 False
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ================= 5. 初始化混合精度与余弦退火 =================
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

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
    total_update_steps = (args.epochs * len(dataloader)) // args.accumulation_steps          
    # 拿出前 10% 的步数做预热 (Warmup)
    warmup_steps = int(total_update_steps * 0.1)           
    # 初始化学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_update_steps
    )

    # ================= 6. checkpoint 检查 =================
    # 尝试加载 Checkpoint
    checkpoint_path = f"{args.checkpoint_dir}/pretrain_checkpoint.pth"
    start_epoch, start_step, swanlab_id = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_path, device, is_distributed
    )

    if is_main_process:
        print(f"使用设备：{device}，是否开启分布式: {is_distributed}")
        print(f"已固定全局随机种子为: {args.seed}")
        if not start_epoch and not start_step:
            print("未找到 checkpoint，开始从零训练...") 
        else:
            print(f"已从 checkpoint 中恢复训练，从第 {start_epoch} 个 epoch 和第 {start_step} 个 step 开始训练...")
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
    # 记录整个训练开始的时间
    start_time = time.time()

    # 如果是恢复训练则从 start_epoch 开始，如果没有 checkpoint 则 start_epoch 默认为0
    for epoch in range(start_epoch, args.epochs):
        # 必须加上这句：让采样器打乱每个 epoch 的数据顺序
        if sampler is not None: 
            sampler.set_epoch(epoch)

        for step, (input_ids, labels) in enumerate(dataloader):
            # 如果是恢复训练，跳过已经练过的 step
            if epoch == start_epoch and step <= start_step:
                continue

            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # DDP 高级优化 防止无意义的梯度同步
            # 在没有达到累积步数时，不需要跨显卡同步梯度，这能极大提升多卡训练速度
            sync_context = model.no_sync if is_distributed and (step + 1) % args.accumulation_steps != 0 else nullcontext

            with sync_context():
                # 前向传播，开启自动混合精度上下文，with 块里的代码会以 fp16 运行
                with torch.autocast(device_type="cuda", dtype=ptdtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss + (outputs.aux_loss if outputs.aux_loss is not None else 0)
                    
                    # 因为 backward() 默认是把梯度加起来，除以步数求平均值，等效于真实的大 Batch
                    loss = loss / args.accumulation_steps

            if step % args.log_interval == 0 and is_main_process:
                print(f"反向传播前真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                # 无论是否更新权重，每一步都要反向传播累积梯度
                scaler.scale(loss).backward()

            if step % args.log_interval == 0 and is_main_process:
                print(f"反向传播后真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"缓存池预留: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader):
                # 梯度裁剪：必须在 unscale 之后进行
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 更新权重和 Scaler
                scaler.step(optimizer)
                scaler.update()
                # 更新学习率 (每个 step 走一步)
                scheduler.step()
                
                # 更新完权重后，立刻清空积攒的梯度，迎接下一轮累积
                optimizer.zero_grad()

            # 3. 记录与上传日志
            if step % args.log_interval == 0 and is_main_process:
                real_loss = loss.item() * args.accumulation_steps
                real_aux_loss = outputs.aux_loss.item() if outputs.aux_loss is not None else 0

                log_training_progress(
                    start_epoch, 
                    start_step,
                    step=step,
                    epoch=epoch,
                    epochs=args.epochs,
                    dataloader_len=len(dataloader),
                    total_steps=args.epochs * len(dataloader),
                    loss_val=real_loss,
                    aux_loss_val=real_aux_loss,
                    lr=optimizer.param_groups[0]['lr'],
                    start_time=start_time,
                    use_swanlab=args.use_swanlab,
                )
                
            # 保存一次 Checkpoint 
            if step > 0 and step % args.save_steps == 0 and is_main_process:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, step, 
                    checkpoint_path, is_distributed, swanlab_id
                )

        # 每轮跑完存一个模型
        if is_main_process:
            # 如果被 DDP 包装过，需要用 model.module.state_dict() 取出真实权重
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(state_dict, f"{args.output_dir}/pretrain_epoch_{epoch}.pth")
            print(f"Epoch {epoch} 完成并保存！")

    if is_main_process and args.use_swanlab:
        swanlab.finish()

    # 销毁分布式进程组
    if is_distributed:
        dist.destroy_process_group()