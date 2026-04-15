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