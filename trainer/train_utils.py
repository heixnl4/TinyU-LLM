import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
import swanlab
import random         # Python 原生随机库
import numpy as np    # NumPy 库
import torch
import torch.distributed as dist
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
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path, is_distributed, swanlab_id):
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
        'swanlab_id': swanlab_id
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint 已保存至: {path} ---")

# Checkpoint 读取函数
def load_checkpoint(model, optimizer, scheduler, scaler, path, device, is_distributed):
    if not os.path.exists(path):
        return 0, 0, None  # 如果文件不存在，从头开始
        
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
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint.get('swanlab_id', None)

# 初始化分布式训练环境，并返回设备信息。
def setup_device_and_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        # 自动选择通信后端 (Linux 用 nccl，Windows 降级为 gloo)
        backend = "nccl" if torch.cuda.is_available() and os.name != 'nt' else "gloo"
        dist.init_process_group(backend=backend)
        
        # 为当前进程绑定对应的显卡
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # 退化为单卡或 CPU 运行
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # 判断是否为主进程（Rank 0），只有主进程才配打日志和存权重
    is_main_process = (local_rank == 0)

    return device, local_rank, is_distributed, is_main_process

# 日志打印函数
def log_training_progress(start_epoch, start_step, step, epoch, epochs, dataloader_len, total_steps, 
                          loss_val, aux_loss_val, lr, start_time, use_swanlab):
    # 1. 计算当前全局进度
    global_step = step + (epoch * dataloader_len)
    calc_step = global_step if global_step > 0 else 1

    # 2. 计算耗时与 ETA
    elapsed_seconds = time.time() - start_time                     
    steps_per_second = calc_step - (start_step + (start_epoch * dataloader_len)) / elapsed_seconds                 
    remaining_steps = total_steps - global_step                    
    eta_seconds = remaining_steps / steps_per_second               

    # 3. 格式化为 HH:MM:SS
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

    # 4. 打印进度日志
    print(f"Epoch: [{epoch+1}/{epochs}] | Step: [{global_step}/{total_steps}] | "
          f"Loss: {loss_val:.4f} | Learning_rate: {lr:.6f} | "
          f"已耗时: {elapsed_str} | 预计剩余: {eta_str}")
    
    # 5. 上传至 SwanLab
    if use_swanlab:
        swanlab.log({
            "train/loss": loss_val,
            "train/aux_loss": aux_loss_val,
            "train/learning_rate": lr,
            "train/step": global_step, 
            "train/elapsed_hours": elapsed_seconds / 3600 
        })