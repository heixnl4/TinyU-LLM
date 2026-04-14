import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import swanlab 
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from model.configuration import TinyuConfig
from dataset.lm_dataset import PretrainDataset 
from transformers import get_cosine_schedule_with_warmup
from trainer.train_utils import print_model_param_details, init_model


# ================= 1. 基础配置 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")
epochs = 2
batch_size = 16
lr = 5e-4

# 初始化 SwanLab：创建一个实验，并记录下这次实验的"超参数"
swanlab.init(
    project="Tinyu-Pretrain",  # 项目名称（相当于一个大文件夹）
    name="simple-test-run-01", # 这次实验的名称（每次跑可以改一下名字对比赛果）
    config={                   # 把你认为重要的配置记下来，方便以后复盘
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "hidden_size": 512,
        "num_layers": 6
    }
)

# ================= 2. 初始化模型与数据 =================
config = TinyuConfig(
    hidden_size=256, 
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    use_moe=True
)

model, tokenizer = init_model(config, device=device)

# 打印模型总参数量（单位：百万，即 M）
print_model_param_details(model, detail=True, prefix="tinyu")


dataset = PretrainDataset("../dataset/pretrain_hq.jsonl", tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=lr)

# ================= 3. 初始化混合精度与余弦退火 =================

# 初始化 GradScaler (用于缩放 fp16 梯度，防止下溢)
scaler = torch.amp.GradScaler("cuda")

# 初始化学习率调度器
total_steps = epochs * len(dataloader)          # 算出一共有多少个 step
warmup_steps = int(total_steps * 0.1)           # 拿出前 10% 的步数做预热 (Warmup)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

# ================= 3. 极简训练循环 =================
model.train()
for epoch in range(epochs):
    for step, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # 前向传播，开启自动混合精度上下文，with 块里的代码会以 fp16 运行
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss + (outputs.aux_loss if outputs.aux_loss is not None else 0)

        if step % 100 == 0:
            print(f"真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"缓存池预留: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # 反向传播，对 loss 进行缩放后再 backward
        scaler.scale(loss).backward()
        # 梯度裁剪：必须在 unscale 之后进行！
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        # 更新权重和 Scaler
        scaler.step(optimizer)
        scaler.update()
        # 更新学习率 (每个 step 走一步)
        scheduler.step()
        
        # 3. 记录与上传日志
        if step % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']  # 获取当前动态学习率
            print(f"Epoch: {epoch}, Step: {step}, Loss: {current_loss:.4f}, LR: {current_lr:.6f}")
            
            # 把当前的数据画到 SwanLab 的折线图上
            swanlab.log({
                "train/loss": current_loss,
                "train/aux_loss": outputs.aux_loss.item() if outputs.aux_loss is not None else 0,
                "train/learning_rate": current_lr,
                "train/step": step + (epoch * len(dataloader)) # 记录全局 step
            })
            
    # 每轮跑完存一个模型
    torch.save(model.state_dict(), f"tinyu_epoch_{epoch}.pth")
    print(f"Epoch {epoch} 完成并保存！")