import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from model.configuration import TinyuConfig
from dataset.lm_dataset import SFTDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_sft_args
from trainer.train_utils import init_model, set_seed, save_checkpoint, load_checkpoint
from trainer.train_utils import setup_device_and_distributed, log_training_progress
from trainer.lora_utils import inject_custom_lora, print_trainable_parameters
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if __name__ == "__main__":
    # ================= 0. 获取全局配置 =================
    args = parse_sft_args()
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_moe{int(args.use_moe)}"
    
    # ================= 1. 初始化分布式环境 =================
    device, local_rank, is_distributed, is_main_process = setup_device_and_distributed()
    set_seed(args.seed + local_rank)

    # ================= 2. 初始化模型并加载预训练权重 =================
    config = TinyuConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )

    model, tokenizer = init_model(config, device=device)

    pretrained_dir = os.path.join(args.output_dir, f"{args.pretrain_run_name}_{arch_signature}")
    if is_main_process:
            os.makedirs(pretrained_dir, exist_ok=True)
    fallback_path = os.path.join(pretrained_dir, "pretrain_weight.pth")

    pretrained_path_to_load = None

    # 优先级 1：用户在命令行明确指定的路径
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        pretrained_path_to_load = args.pretrained_model_path
        if is_main_process:
            print(f" 找到指定路径的预训练权重: {pretrained_path_to_load}")
            
    # 优先级 2：找默认的 fallback 路径
    elif os.path.exists(fallback_path):
        pretrained_path_to_load = fallback_path
        if is_main_process:
            print(f" 未找到指定权重，使用备用路径的预训练权重: {pretrained_path_to_load}")
            
    # 优先级 3
    else:
        if is_main_process:
            print(f"警告: 指定路径 [{args.pretrained_model_path}] 和备用路径 [{fallback_path}] 均未找到权重文件！")
            print(f"正在对【随机初始化】的 {arch_signature} 模型进行 SFT (仅供 Debug)！")

    # 执行最终的加载动作
    if pretrained_path_to_load:
        # 注意 map_location 必须传当前卡，防止多卡加载时显存爆炸
        state_dict = torch.load(pretrained_path_to_load, map_location=device)
        
        # 兼容处理：剥离预训练保存时可能带有的 DDP "module." 前缀
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
        # DDP 遇到冻结参数时，偶尔会报错，需要开启 find_unused_parameters=False
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ================= 4. 数据集与优化器 =================
    dataset = SFTDataset(args.data_path, tokenizer, args.max_length)

    if is_distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 关键：优化器只接收需要计算梯度的参数 (即 LoRA 的参数)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # ================= 5. 初始化混合精度与余弦退火 =================
    ptdtype = torch.float16
    if args.dtype == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            ptdtype = torch.bfloat16
            if is_main_process: print("硬件支持 BF16，已开启 Bfloat16 混合精度训练")
        else:
            if is_main_process: print("当前硬件不支持 BF16，已自动降级为 Float16")
            ptdtype = torch.float16
    elif args.dtype == "float32":
        ptdtype = torch.float32

    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))
    total_update_steps = (args.epochs * len(dataloader)) // args.accumulation_steps          
    warmup_steps = int(total_update_steps * 0.1)           
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_update_steps
    )

    # ================= 6. checkpoint 检查 =================
    current_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.run_name}_{arch_signature}")
    checkpoint_path = f"{current_ckpt_dir}/sft_checkpoint.pth"
    start_epoch, start_step, swanlab_id = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_path, device, is_distributed, strict=False
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
        swanlab_id = swanlab_id if swanlab_id else None
        resume = 'must' if swanlab_id else None
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
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None: sampler.set_epoch(epoch)

        for step, (input_ids, labels) in enumerate(dataloader):
            if epoch == start_epoch and step <= start_step:
                continue
            input_ids, labels = input_ids.to(device), labels.to(device)
            # 判断当前 step 是否是累积的最后一步
            is_update_step = (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader)
            
            sync_context = model.no_sync if is_distributed and (step + 1) % args.accumulation_steps != 0 else nullcontext
            with sync_context():
                with torch.autocast(device_type="cuda", dtype=torch.float16): # 或 ptdtype
                    # 这里的 labels 已经被 SFTDataset 处理过了 (Prompt 部分是 -100)
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss + outputs.aux_loss
                    loss = loss / args.accumulation_steps

                if step % args.log_interval == 0 and is_main_process:
                    print(f"反向传播前真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                # 无论是否更新权重，每一步都要反向传播累积梯度
                scaler.scale(loss).backward()

                if step % args.log_interval == 0 and is_main_process:
                    print(f"反向传播后真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    print(f"缓存池预留: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

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
                    checkpoint_path, is_distributed, swanlab_id,
                    only_lora=True  # 必须开启过滤
                )

        # ================= 6. 保存 LoRA 权重 =================
        if is_main_process:
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
            os.makedirs(output_dir, exist_ok=True)
            # SFT 保存模型时，只保存 LoRA 的权重 
            lora_state = {k: v for k, v in state_dict.items() if 'lora_A' in k or 'lora_B' in k}
            if epoch == (args.epochs - 1):
                torch.save(state_dict, f"{output_dir}/lora_weight.pth")
            else:
                torch.save(state_dict, f"{output_dir}/lora_epoch_{epoch}.pth")
            print(f"Epoch {epoch} 完成，LoRA 权重已保存至 {output_dir}")
                        
    if is_main_process and args.use_swanlab:
        swanlab.finish()

    if is_distributed:
        dist.destroy_process_group()