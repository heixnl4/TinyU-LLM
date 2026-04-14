'''
该文件做单卡分布式测试使用，需要对代码做临时修改：
由于只有一张卡，当 torchrun 拉起第二个进程时，它的 local_rank 会变成 1。如果代码按原逻辑执行 torch.cuda.set_device(1)，程序会直接报错“找不到设备”。
标记🌟的部分是与原文件有改动的部分
运行方法：
第一步：如果是windows系统，临时关闭 libuv（只在当前窗口生效）：
$env:USE_LIBUV=0
第二步：运行分布式代码
torchrun --nproc_per_node=2 train_pretrain.py
'''
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

# 导入分布式计算需要的模块
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ================= 1. 初始化分布式环境 =================
# 1.1 从环境变量中获取当前进程的 local_rank（torchrun 会自动注入这个变量）
local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_distributed = local_rank != -1

if is_distributed:
    # 1.2 初始化进程组 (Linux 用 nccl，Windows 通常用 gloo，但推荐在 WSL 里用 nccl)
    backend = "nccl" if torch.cuda.is_available() and os.name != 'nt' else "gloo"
    dist.init_process_group(backend=backend)
    
    # 🌟 修改点 1：不管 local_rank 是几，强行把它们都按在第 0 张卡上！
    torch.cuda.set_device(0) 
    device = torch.device("cuda:0")
else:
    # 退化为单卡运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0

# 判断当前是否为主进程 (用于打印日志和保存模型，防止多卡重复输出)
is_main_process = (local_rank == 0)
if is_main_process:
    print(f"使用设备：{device}，是否开启分布式: {is_distributed}")


epochs = 2
# 🌟防止显存爆炸，batch_size改为1
batch_size = 1
lr = 5e-4
use_compile = False  # 在 Windows 原生系统保持 False，Linux 下可改为 True。

if is_main_process:
    # 初始化 SwanLab：创建一个实验，并记录下这次实验的"超参数"
    swanlab.init(
        project="Tinyu-Pretrain",  # 项目名称（相当于一个大文件夹）
        name="simple-test-run", # 这次实验的名称
        config={                   # 把你认为重要的配置记下来，方便以后复盘
            "epochs": epochs,
            "batch_size": batch_size,
            "use_compile": use_compile,
            "learning_rate": lr,
            "hidden_size": 64,
            "num_layers": 6
        }
    )

# ================= 2. 初始化模型与数据 =================
config = TinyuConfig(
    hidden_size=256, 
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    use_moe=False
)

model, tokenizer = init_model(config, device=device)

# 打印模型总参数量（单位：百万，即 M）
if is_main_process:
    print_model_param_details(model, detail=False)

# ================= 4. 模型编译与 DDP 包装 =================
# 4.1 模型编译 (必须在 DDP 之前进行)
if use_compile:
    if is_main_process: print("正在使用 torch.compile 编译模型...")
    model = torch.compile(model)

# 4.2 DDP 包装
if is_distributed:
    # 告诉 DDP 忽略掉 RoPE 的缓存参数，否则会报错
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    # 🌟 修改点 2：强行指定 device_ids 为 [0]
    model = DDP(model, device_ids=[0])


# ================= 5. 数据集与 DistributedSampler =================
dataset = PretrainDataset("../dataset/pretrain_hq.jsonl", tokenizer, max_length=512)

if is_distributed:
    # 分布式采样器：确保不同的 GPU 拿到不同的数据切片，不会重复训练
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False) # 注意：用 sampler 时 shuffle 必须为 False
else:
    sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ================= 3. 初始化混合精度与余弦退火 =================
optimizer = optim.AdamW(model.parameters(), lr=lr)
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

# ================= 7. 训练循环 =================
model.train()
for epoch in range(epochs):
    # 必须加上这句：让采样器打乱每个 epoch 的数据顺序
    if sampler is not None: 
        sampler.set_epoch(epoch)

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
        if step % 100 == 0 and is_main_process:
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
    # 只允许主进程保存，防止多个 GPU 抢占覆盖同一个文件
    if is_main_process:
        # 如果被 DDP 包装过，需要用 model.module.state_dict() 取出真实权重
        state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save(state_dict, f"tinyu_epoch_{epoch}.pth")
        print(f"Epoch {epoch} 完成并保存！")

if is_main_process:
    swanlab.finish()

# 销毁分布式进程组
if is_distributed:
    dist.destroy_process_group()