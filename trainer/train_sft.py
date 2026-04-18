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

# 导入 PEFT (LoRA) 相关库
from peft import LoraConfig, get_peft_model, TaskType

from model.configuration import TinyuConfig
# 注意：这里需要你新建一个 SFTDataset (稍后提供代码)
from dataset.sft_dataset import SFTDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.arguments import parse_args
from trainer.train_utils import init_model, set_seed, save_checkpoint, load_checkpoint
from trainer.train_utils import setup_device_and_distributed, log_training_progress

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if __name__ == "__main__":
    args = parse_args()
    device, local_rank, is_distributed, is_main_process = setup_device_and_distributed()
    set_seed(args.seed + local_rank)

    # ================= 1. 初始化模型并加载预训练权重 =================
    config = TinyuConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )
    # 此处的 init_model 内部应该是随机初始化的
    model, tokenizer = init_model(config, device=device)
    
    # 🌟 关键：在注入 LoRA 前，必须先加载你跑出来的预训练权重！
    # 假设你命令行传入了 --pretrained_model_path ./tinyu_out/pretrain_epoch_1.pth
    if hasattr(args, 'pretrained_model_path') and args.pretrained_model_path:
        state_dict = torch.load(args.pretrained_model_path, map_location=device)
        # 剥离可能存在的 DDP 前缀
        clean_state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
        if is_main_process: print(f"✅ 成功加载预训练基座权重: {args.pretrained_model_path}")
    else:
        if is_main_process: print("⚠️ 警告：未提供预训练权重，正在对随机初始化的模型进行 SFT (仅供 Debug)！")

    # ================= 2. 注入 LoRA =================
    # ⚠️ 避坑：target_modules 必须和你手写 TinyuModel 里的 nn.Linear 变量名完全一致！
    # 如果你的 Attention 层里叫 self.wq, self.wk，这里就要写 ["wq", "wk"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,               # LoRA 的秩 (Rank)，通常设为 8, 16, 64
        lora_alpha=32,     # 缩放因子，通常为 r 的两倍或四倍
        lora_dropout=0.1,  # 防止过拟合的丢弃率
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # 需要注入 LoRA 的层
    )
    
    # 这一步会冻结原模型的所有参数，并添加可训练的 A/B 矩阵
    model = get_peft_model(model, peft_config)
    
    if is_main_process:
        # 打印一下可训练参数的比例 (通常在 0.1% ~ 1% 之间)
        model.print_trainable_parameters()

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

    # 🌟 关键：优化器只接收需要计算梯度的参数 (即 LoRA 的参数)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # ... (混合精度 ptdtype、scaler、scheduler 的初始化逻辑与之前完全一致，此处省略以保持精简) ...

    # ================= 5. SFT 训练循环 =================
    model.train()
    optimizer.zero_grad()
    start_time = time.time()

    for epoch in range(args.epochs):
        if sampler is not None: sampler.set_epoch(epoch)

        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            sync_context = model.no_sync if is_distributed and (step + 1) % args.accumulation_steps != 0 else nullcontext

            with sync_context():
                with torch.autocast(device_type="cuda", dtype=torch.float16): # 或 ptdtype
                    # 这里的 labels 已经被 SFTDataset 处理过了 (Prompt 部分是 -100)
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss + (outputs.aux_loss if outputs.aux_loss is not None else 0)
                    loss = loss / args.accumulation_steps

                scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                # 裁剪梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # ... (日志记录逻辑与之前完全一致) ...

        # ================= 6. 保存 LoRA 权重 =================
        if is_main_process:
            # 🌟 极度重要：SFT 保存模型时，只保存 LoRA 的权重 (通常只有几十MB)！
            # 绝对不要把冻结的几十GB的基座权重又存一遍。
            save_dir = f"{args.output_dir}/sft_epoch_{epoch}"
            os.makedirs(save_dir, exist_ok=True)
            
            # 提取 PEFT 模型并保存
            peft_model = model.module if is_distributed else model
            peft_model.save_pretrained(save_dir)
            print(f"Epoch {epoch} 完成，LoRA 权重已保存至 {save_dir}！")

    if is_distributed:
        dist.destroy_process_group()