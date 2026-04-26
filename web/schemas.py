"""
FastAPI Pydantic 数据模型定义
用于请求参数校验和响应序列化
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class DType(str, Enum):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


class TaskStatus(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"
    stopped = "stopped"


# ==================== 预训练参数 ====================
class PretrainConfig(BaseModel):
    epochs: int = Field(default=2, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=5e-4, gt=0, le=1e-1)
    seed: int = Field(default=42)
    max_length: int = Field(default=512, ge=64, le=8192)
    grad_clip: float = Field(default=1.0, ge=0.1, le=10.0)
    accumulation_steps: int = Field(default=4, ge=1, le=64)
    dtype: DType = Field(default=DType.bfloat16)

    # 模型架构
    hidden_size: int = Field(default=512, ge=128, le=4096)
    num_hidden_layers: int = Field(default=4, ge=1, le=32)
    num_attention_heads: int = Field(default=8, ge=1, le=64)
    num_key_value_heads: int = Field(default=2, ge=1, le=64)
    use_moe: bool = Field(default=False)
    use_compile: bool = Field(default=False)
    use_swanlab: bool = Field(default=False)

    # 路径
    data_path: str = Field(default="./dataset/pretrain_hq.jsonl")
    checkpoint_dir: str = Field(default="./checkpoints")
    output_dir: str = Field(default="./out")
    save_steps: int = Field(default=1000, ge=1)

    # 日志
    log_interval: int = Field(default=100, ge=1)
    project_name: str = Field(default="TinyU-LLM-Pretrain")
    run_name: str = Field(default="run-web")


# ==================== SFT / LoRA 参数 ====================
class SFTConfig(BaseModel):
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=5e-5, gt=0, le=1e-1)
    seed: int = Field(default=42)
    max_length: int = Field(default=512, ge=64, le=8192)
    grad_clip: float = Field(default=1.0, ge=0.1, le=10.0)
    accumulation_steps: int = Field(default=4, ge=1, le=64)
    dtype: DType = Field(default=DType.bfloat16)

    # 模型架构
    hidden_size: int = Field(default=512, ge=128, le=4096)
    num_hidden_layers: int = Field(default=4, ge=1, le=32)
    num_attention_heads: int = Field(default=8, ge=1, le=64)
    num_key_value_heads: int = Field(default=2, ge=1, le=64)
    use_moe: bool = Field(default=False)
    use_compile: bool = Field(default=False)
    use_swanlab: bool = Field(default=False)

    # 路径
    data_path: str = Field(default="./dataset/sft_mini_512_part.jsonl")
    checkpoint_dir: str = Field(default="./checkpoints")
    output_dir: str = Field(default="./out")
    save_steps: int = Field(default=10, ge=1)

    # 日志
    log_interval: int = Field(default=5, ge=1)
    project_name: str = Field(default="TinyU-LLM-SFT")
    run_name: str = Field(default="lora-run-web")
    pretrain_run_name: str = Field(default="run-web")

    # LoRA 参数
    pretrained_model_path: Optional[str] = Field(default=None)
    lora_rank: int = Field(default=8, ge=1, le=128)
    lora_alpha: float = Field(default=32.0, ge=1.0, le=256.0)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    target_modules: List[str] = Field(default=["q_proj", "k_proj", "v_proj", "o_proj"])


# ==================== 对话参数 ====================
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=0, le=100)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    response: str


# ==================== 权重加载参数 ====================
class LoadWeightRequest(BaseModel):
    weight_path: str = Field(..., description="权重文件路径 (.pth)")
    hidden_size: int = Field(default=512)
    num_hidden_layers: int = Field(default=4)
    num_attention_heads: int = Field(default=8)
    num_key_value_heads: int = Field(default=2)
    use_moe: bool = Field(default=False)
    lora_path: Optional[str] = Field(default=None, description="LoRA 权重路径（可选）")


# ==================== 通用响应 ====================
class ApiResponse(BaseModel):
    code: int = Field(default=0)
    message: str = Field(default="success")
    data: Optional[dict] = None


class TaskInfo(BaseModel):
    task_id: str
    task_type: str
    status: TaskStatus
    config: Optional[dict] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
