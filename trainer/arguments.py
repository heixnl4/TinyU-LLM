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


def parse_ppo_args():
    parser = argparse.ArgumentParser(description="TinyU-LLM PPO 训练全局参数配置")

    # ================= 1. 基础训练与 PPO 超参数 =================
    parser.add_argument("--epochs", type=int, default=1, help="外层经验收集(Rollout)的总轮数")
    parser.add_argument("--ppo_epochs", type=int, default=1, help="PPO内层使用同一批数据更新的轮数")
    parser.add_argument("--rollout_batch_size", type=int, default=1, help="单卡用于生成经验(Rollout)的 Batch Size")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="PPO阶段的梯度累积步数")
    
    # 学习率与优化
    parser.add_argument("--actor_learning_rate", type=float, default=1e-5, help="Actor(策略)模型最大学习率，通常较小")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-5, help="Critic(价值)模型最大学习率，通常较大")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="训练时使用的精度")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    
    # 长度控制
    parser.add_argument("--max_prompt_length", type=int, default=256, help="输入 Prompt 的最大长度")
    parser.add_argument("--max_response_length", type=int, default=256, help="Actor 生成回复的最大长度")

    # PPO 专属核心超参数
    parser.add_argument("--gamma", type=float, default=0.99, help="奖励折扣因子(Discount factor)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="广义优势估计(GAE)的 lambda 参数")
    parser.add_argument("--cliprange", type=float, default=0.2, help="Actor 策略更新的截断比率 (PPO clip)")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Critic 价值更新的截断比率 (Value clip)")

    # ================= 2. 模型架构参数 =================
    parser.add_argument("--hidden_size", type=int, default=64, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="隐藏层层数")
    parser.add_argument("--num_attention_heads", type=int, default=2, help="注意力头数")
    parser.add_argument("--num_key_value_heads", type=int, default=2, help="KV 头数")
    
    parser.add_argument("--use_moe", action="store_true", help="是否开启 MoE 架构")
    parser.add_argument("--use_compile", action="store_true", help="是否开启 torch.compile 加速")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用Swanlab")

    # ================= 3. 路径与保存配置 =================
    # PPO 专属路径组
    parser.add_argument("--data_path", type=str, default="../dataset/test_data.jsonl", help="PPO Prompt 训练数据集路径")
    parser.add_argument("--actor_model_path", type=str, default="../out/sft_model.pth", help="SFT模型权重路径(用于初始化Actor和Reference)")
    parser.add_argument("--reward_model_path", type=str, default="../out/reward_model.pth", help="奖励模型权重路径(用于初始化Critic和Reward Model)")
    
    # 常规保存路径
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="PPO Checkpoint 保存目录")
    parser.add_argument("--output_dir", type=str, default="../out", help="PPO 最终权重保存目录")
    parser.add_argument("--save_steps", type=int, default=1, help="每隔多少次更新保存一次 Checkpoint")

    # ================= 4. 日志与实验追踪 =================
    parser.add_argument("--log_interval", type=int, default=1, help="每隔多少步打印一次日志(PPO更新步数)")
    parser.add_argument("--project_name", type=str, default="TinyU-LLM-PPO", help="SwanLab 项目名称")
    parser.add_argument("--run_name", type=str, default="ppo-run-1", help="本次实验名称")

    args = parser.parse_args()
    
    # 可以在这里做一些简单的 PPO 参数合法性校验
    # 例如将多个模型路径组装成字典方便后续调用
    args.model_paths = {
        "actor_sft": args.actor_model_path,
        "reward": args.reward_model_path
    }
    
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

    parser.add_argument("--run_name", type=str, default="run-2", help="本次实验名称")

    args = parser.parse_args()
    return args


# ================== Web 后端支持：从字典构建 Namespace ==================
import argparse

def _dict_to_namespace(data: dict, defaults: dict) -> argparse.Namespace:
    """
    将前端传来的字典与默认值合并，构建 argparse.Namespace。
    注意：action='store_true' 的布尔字段，字典中传 None 或 False 时不应触发。
    """
    merged = dict(defaults)
    merged.update(data)
    ns = argparse.Namespace()
    for k, v in merged.items():
        setattr(ns, k, v)
    return ns


def build_pretrain_args_from_dict(data: dict) -> argparse.Namespace:
    """从前端配置字典构建预训练参数对象。"""
    defaults = {
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "seed": 42,
        "max_length": 512,
        "grad_clip": 1.0,
        "accumulation_steps": 4,
        "dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "use_moe": False,
        "use_compile": False,
        "use_swanlab": False,
        "data_path": "./dataset/pretrain_hq.jsonl",
        "checkpoint_dir": "./checkpoints",
        "output_dir": "./out",
        "save_steps": 1000,
        "log_interval": 100,
        "project_name": "TinyU-LLM-Pretrain",
        "run_name": "run-web",
    }
    return _dict_to_namespace(data, defaults)


def build_sft_args_from_dict(data: dict) -> argparse.Namespace:
    """从前端配置字典构建 SFT 参数对象。"""
    defaults = {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "seed": 42,
        "max_length": 512,
        "grad_clip": 1.0,
        "accumulation_steps": 4,
        "dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "use_moe": False,
        "use_compile": False,
        "use_swanlab": False,
        "data_path": "./dataset/sft_mini_512_part.jsonl",
        "checkpoint_dir": "./checkpoints",
        "output_dir": "./out",
        "save_steps": 10,
        "log_interval": 5,
        "project_name": "TinyU-LLM-SFT",
        "run_name": "lora-run-web",
        "pretrain_run_name": "run-web",
        "pretrained_model_path": None,
        "lora_rank": 8,
        "lora_alpha": 32.0,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    return _dict_to_namespace(data, defaults)