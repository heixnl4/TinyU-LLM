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
from torch.utils.data import Sampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from model.model_Tinyu import TinyuForcausalLM

# 打印模型参数
def print_model_param_details(model, detail=False, prefix=""):
    print(f"{'-'*50}")
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_count = param.numel()
        total_params += param_count
        if detail:
            print(f"{prefix}{name}: {param_count / 1e6:.3f}M")
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
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path, is_distributed, swanlab_id, only_lora=False, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 如果是分布式训练，需要取 model.module
    model_state = model.module.state_dict() if is_distributed else model.state_dict()

    # 仅保存 LoRA 参数
    if only_lora:
        lora_state = {k: v for k, v in model_state.items() if 'lora_A' in k or 'lora_B' in k}
        model_state = lora_state
        print(f"已过滤出 {len(model_state)} 个 LoRA 参数进行保存。")
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'swanlab_id': swanlab_id
    }

    # 其他参数，如ppo中的critic模型
    for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    checkpoint[key] = raw_value.state_dict()
                else:
                    checkpoint[key] = value

    torch.save(checkpoint, path)
    print(f"--- Checkpoint 已保存至: {path} ---")

# Checkpoint 读取函数
def load_checkpoint(model, optimizer, scheduler, scaler, path, device, is_distributed, strict=True, **kwargs):
    if not os.path.exists(path):
        return -1, -1, None
        
    print(f"--- 正在从 Checkpoint 恢复: {path} ---")
    checkpoint = torch.load(path, map_location=device)
    # ================= 1. 恢复基础组件 =================
    # 核心加载逻辑：如果是 SFT 恢复，strict 必须传 False
    # strict=False 只加载字典里有的，没找到的保持刚加载的预训练基座权重”。
    if is_distributed:
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # ================= 2. 动态恢复 kwargs 里的附加组件 =================
    for key, obj in kwargs.items():
        if key in checkpoint:
            if hasattr(obj, 'load_state_dict'):
                # 剥离 DDP 包装（如果传入的是个被包裹的模型）
                raw_obj = obj.module if isinstance(obj, DistributedDataParallel) else obj
                
                # 模型 (nn.Module) 的 load_state_dict 支持 strict 参数，而优化器/调度器不支持
                if isinstance(raw_obj, torch.nn.Module):
                    raw_obj.load_state_dict(checkpoint[key], strict=strict)
                else:
                    raw_obj.load_state_dict(checkpoint[key])
                    
                print(f"已动态恢复附加状态: {key}")
            else:
                # 对于非 state_dict 对象（如布尔值、字符串），无法就地更新
                print(f"警告: '{key}' 没有 load_state_dict 方法，跳过动态加载。")
        else:
            print(f"警告: Checkpoint 文件中未找到 '{key}'，跳过加载。")
    
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
                          loss_val, aux_loss_val, lr, start_time, use_swanlab, reward_acc=None):
    # 1. 计算当前全局进度
    global_step = step + 1 + (epoch * dataloader_len)
    calc_step = global_step if global_step > 0 else 1

    # 2. 计算耗时与 ETA
    elapsed_seconds = time.time() - start_time
    steps_per_second = (calc_step - (start_step + (start_epoch * dataloader_len))) / elapsed_seconds
    remaining_steps = total_steps - global_step
    eta_seconds = remaining_steps / steps_per_second

    # 3. 格式化为 HH:MM:SS
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

    # 4. 打印进度日志
    acc_str = f" | Reward Acc: {reward_acc:.2%}" if reward_acc is not None else ""
    print(f"Epoch: [{epoch+1}/{epochs}] | Step: [{global_step}/{total_steps}] | "
          f"Loss: {loss_val:.4f} | Learning_rate: {lr:.6f}{acc_str} | "
          f"本次训练已耗时: {elapsed_str} | 预计剩余: {eta_str}")

    # 5. 上传至 SwanLab
    if use_swanlab:
        log_dict = {
            "train/loss": loss_val,
            "train/aux_loss": aux_loss_val,
            "train/learning_rate": lr,
            "train/step": global_step,
            "train/elapsed_hours": elapsed_seconds / 3600
        }
        if reward_acc is not None:
            log_dict["train/reward_acc"] = reward_acc
        swanlab.log(log_dict)


class SkipStepSampler(Sampler):
    """
    一个用于断点恢复的 Sampler 包装器。
    它会在迭代时直接截断掉前 N 个 step 对应的索引，从而完美避开数据的硬盘 IO 读取。
    """
    def __init__(self, base_sampler, skip_steps, batch_size):
        self.base_sampler = base_sampler
        # 换算成需要跳过的样本总数 (每张卡单独跳过自己的部分)
        self.skip_samples = skip_steps * batch_size

    def __iter__(self):
        # 拿到原始 Sampler 排好序/打乱后的所有索引
        indices = list(iter(self.base_sampler))
        
        # 核心：直接在索引列表上进行切片截断
        if self.skip_samples >= len(indices):
            return iter([])
            
        return iter(indices[self.skip_samples:])

    def __len__(self):
        return max(0, len(self.base_sampler) - self.skip_samples)

    def set_epoch(self, epoch):
        # 兼容 DistributedSampler 的打乱机制
        if hasattr(self.base_sampler, 'set_epoch'):
            self.base_sampler.set_epoch(epoch)