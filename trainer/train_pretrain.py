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
from model.model_Tinyu import TinyuForcausalLM 
from dataset.lm_dataset import PretrainDataset 
from trainer.train_utils import print_model_param_details, init_model
from transformers import AutoTokenizer


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
) # 先用小参数测试
trainer_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(trainer_dir, '..', 'model'))
model = TinyuForcausalLM(config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 打印模型总参数量（单位：百万，即 M）
print_model_param_details(model, detail=True, prefix="tinyu")


dataset = PretrainDataset("../dataset/pretrain_hq.jsonl", tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=lr)

# ================= 3. 极简训练循环 =================
model.train()
for epoch in range(epochs):
    for step, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # 1. 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss + (outputs.aux_loss if outputs.aux_loss is not None else 0)
        
        # 2. 反向传播,梯度裁剪,更新参数
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # 3. 记录与上传日志
        if step % 10 == 0:
            current_loss = loss.item()
            print(f"Epoch: {epoch}, Step: {step}, Loss: {current_loss:.4f}")
            
            # 把当前的数据画到 SwanLab 的折线图上
            swanlab.log({
                "train/loss": current_loss,
                "train/aux_loss": outputs.aux_loss.item() if outputs.aux_loss is not None else 0,
                "train/step": step + (epoch * len(dataloader)) # 记录全局 step
            })
            
    # 每轮跑完存一个模型
    torch.save(model.state_dict(), f"tinyu_epoch_{epoch}.pth")
    print(f"Epoch {epoch} 完成并保存！")