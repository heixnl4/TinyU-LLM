import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random         # Python 原生随机库
import numpy as np    # NumPy 库
import torch
from transformers import AutoTokenizer
from model.model_Tinyu import TinyuForcausalLM

# 打印模型参数
def print_model_param_details(model, detail=False, prefix=""):
    print("=== Detailed Parameter Count ===")
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_count = param.numel()
        total_params += param_count
        if detail:
            print(f"{prefix}{name}: {param_count / 1e6:.3f}M")
    print(f"{'-'*50}")
    print(f"Total trainable params: {total_params / 1e6:.2f}M")
    print(f"{'-'*50}")

# 初始化模型和tokenizer
def init_model(config, tokenizer_path='../model', device='cuda'):
    model = TinyuForcausalLM(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer

# 定义随机种子固定函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 针对多卡环境
    
    # 开启 cuDNN 的确定性行为（会牺牲极其微小的一点点训练速度，但保证了100%可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Checkpoint 保存函数
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path, is_distributed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 如果是分布式训练，需要取 model.module
    model_state = model.module.state_dict() if is_distributed else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint 已保存至: {path} ---")

# Checkpoint 读取函数
def load_checkpoint(model, optimizer, scheduler, scaler, path, device, is_distributed):
    if not os.path.exists(path):
        return 0, 0  # 如果文件不存在，从头开始
        
    print(f"--- 正在从 Checkpoint 恢复: {path} ---")
    checkpoint = torch.load(path, map_location=device)
    
    # 加载权重
    if is_distributed:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step']