# trainer/arguments.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TinyU-LLM 训练全局参数配置")

    # ================= 1. 基础训练参数 =================
    parser.add_argument("--epochs", type=int, default=2, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="单卡 Batch Size")
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
    parser.add_argument("--num_key_value_heads", type=int, default=4, help="KV 头数")
    
    # 布尔类型的开关参数，如果在命令行加上 --use_moe 就是 True，不加就是 False
    parser.add_argument("--use_moe", action="store_true", help="是否开启 MoE 架构")
    parser.add_argument("--use_compile", action="store_true", help="是否开启 torch.compile 加速")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用Swanlab")

    # ================= 3. 路径与保存配置 =================
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_part.jsonl", help="训练数据集路径")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="模型与 Checkpoint 保存目录")
    parser.add_argument("--output_dir", type=str, default="../out", help="模型与 Checkpoint 保存目录")
    parser.add_argument("--save_steps", type=int, default=1000, help="每隔多少步保存一次 Checkpoint")

    # ================= 4. 日志与实验追踪 =================
    parser.add_argument("--log_interval", type=int, default=100, help="每隔多少步打印一次日志")
    parser.add_argument("--project_name", type=str, default="TinyU-LLM-Pretrain", help="SwanLab 项目名称")
    parser.add_argument("--run_name", type=str, default="run-default", help="本次实验名称")

    args = parser.parse_args()
    return args