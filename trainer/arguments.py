# trainer/arguments.py
import argparse

def parse_pretrain_args():
    parser = argparse.ArgumentParser(description="TinyU-LLM 训练全局参数配置")

    # ================= 1. 基础训练参数 =================
    parser.add_argument("--epochs", type=int, default=2, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="单卡 Batch Size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="最大学习率")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--max_length", type=int, default=512, help="模型最大上下文长度")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="训练时使用的精度")
    
    # ================= 2. 模型架构参数 =================
    # (预训练从零开始时需要，如果是 SFT 微调则可能直接从预训练权重读取，这里留作覆盖用)
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="隐藏层层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_key_value_heads", type=int, default=2, help="KV 头数")
    
    # 布尔类型的开关参数，如果在命令行加上 --use_moe 就是 True，不加就是 False
    parser.add_argument("--use_moe", action="store_true", help="是否开启 MoE 架构")
    parser.add_argument("--use_compile", action="store_true", help="是否开启 torch.compile 加速")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用Swanlab")

    # ================= 3. 路径与保存配置 =================
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="训练数据集路径")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="模型与 Checkpoint 保存目录")
    parser.add_argument("--output_dir", type=str, default="../out", help="模型与 Checkpoint 保存目录")
    parser.add_argument("--save_steps", type=int, default=1000, help="每隔多少步保存一次 Checkpoint")

    # ================= 4. 日志与实验追踪 =================
    parser.add_argument("--log_interval", type=int, default=100, help="每隔多少步打印一次日志")
    parser.add_argument("--project_name", type=str, default="TinyU-LLM-Pretrain", help="SwanLab 项目名称")
    parser.add_argument("--run_name", type=str, default="run-2", help="本次实验名称")

    args = parser.parse_args()
    return args

def parse_sft_args():
    parser = argparse.ArgumentParser(description="TinyU-LLM SFT 微调全局参数配置")

    # ================= 1. 基础训练参数 =================
    parser.add_argument("--epochs", type=int, default=3, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="单卡 Batch Size")
    # SFT 的学习率通常要比预训练小，比如 5e-5 或 1e-4，防止产生灾难性遗忘
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="最大学习率")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--max_length", type=int, default=512, help="模型最大上下文长度")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="训练时使用的精度")
    
    # ================= 2. 模型架构参数 =================
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="隐藏层层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_key_value_heads", type=int, default=2, help="KV 头数")
    parser.add_argument("--use_moe", action="store_true", help="是否开启 MoE 架构")
    parser.add_argument("--use_compile", action="store_true", help="是否开启 torch.compile 加速")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用 Swanlab")

    # ================= 3. 路径与保存配置 =================
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512_part.jsonl", help="SFT 训练数据集路径")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="SFT 断点保存目录")
    parser.add_argument("--output_dir", type=str, default="../out", help="LoRA 权重最终保存目录")
    parser.add_argument("--save_steps", type=int, default=10, help="每隔多少步保存一次 Checkpoint")

    # ================= 4. 日志与实验追踪 =================
    parser.add_argument("--log_interval", type=int, default=5, help="每隔多少步打印一次日志")
    parser.add_argument("--project_name", type=str, default="TinyU-LLM-SFT", help="SwanLab 项目名称 (SFT阶段)")
    parser.add_argument("--run_name", type=str, default="lore-run-1", help="本次实验名称")
    parser.add_argument("--pretrain_run_name", type=str, default="run-1", help="本次实验名称")

    # ================= 5. SFT & LoRA 专属参数 =================
    # 必须提供基座权重路径，否则模型就是在对随机初始化的乱码进行微调
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="预训练基座模型权重路径 (.pth 文件)")
    
    # LoRA 核心四参数
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA 的秩 (Rank)，决定了外挂参数量的大小")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA 缩放因子 (通常设为 rank 的 2-4 倍)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA 层的 Dropout 概率")
    
    # 使用 nargs="+" 允许在命令行传入列表，例如: --target_modules q_proj v_proj
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"], help="需要注入 LoRA 的目标线性层")

    args = parser.parse_args()
    return args


def inference_args():
    parser = argparse.ArgumentParser(description="TinyU-LLM 训练全局参数配置")

    parser.add_argument("--epochs", type=int, default=2, help="训练总轮数")
    
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="隐藏层层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_key_value_heads", type=int, default=2, help="KV 头数")
    
    parser.add_argument("--use_moe", action="store_true", help="是否开启 MoE 架构")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型与 Checkpoint 保存目录")
    parser.add_argument("--output_dir", type=str, default="./out", help="模型与 Checkpoint 保存目录")

    parser.add_argument("--run_name", type=str, default="run-1", help="本次实验名称")

    args = parser.parse_args()
    return args