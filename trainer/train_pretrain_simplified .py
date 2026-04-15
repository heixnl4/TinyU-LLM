import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torch import optim
from model.configuration import TinyuConfig
from model.model_Tinyu import TinyuForcausalLM 
from dataset.lm_dataset import PretrainDataset 
from transformers import AutoTokenizer

# ================= 1. 基础配置 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 2
batch_size = 16
lr = 5e-4

# ================= 2. 初始化模型与数据 =================
config = TinyuConfig(
    hidden_size=256, 
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    use_moe=True
)

tokenizer = AutoTokenizer.from_pretrained("../model")
model = TinyuForcausalLM(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
dataset = PretrainDataset("../dataset/pretrain_hq.jsonl", tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ================= 3. 极简训练循环 =================
model.train()
for epoch in range(epochs):
    for step, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # 1. 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss + (outputs.aux_loss if outputs.aux_loss is not None else 0)
        
        # 2. 反向传播
        loss.backward()
        
        # 3. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 4. 更新权重
        optimizer.step()
        optimizer.zero_grad()
        
        # 打印日志
        if step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")
            
    # 每轮跑完存一个模型
    os.makedirs("../out", exist_ok=True)
    torch.save(model.state_dict(), f"../out/pretrain_s_epoch_{epoch}.pth")
    print(f"Epoch {epoch} 完成并保存！")