"""
对话推理引擎
管理模型的单例加载、权重导入和流式生成。
"""
import os
import sys
import re
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Optional, Iterator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.configuration import TinyuConfig
from trainer.train_utils import init_model, print_model_param_details
from web.schemas import LoadWeightRequest


class ChatEngine:
    """
    单例聊天引擎。模型加载后常驻显存，避免重复初始化。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.config = None
        self.weight_info = {
            "base_weight": None,
            "lora_weight": None,
            "loaded_at": None,
        }

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self, req: LoadWeightRequest) -> dict:
        """
        根据请求加载模型和权重。
        返回加载结果信息字典。
        """
        self.device = self._detect_device()

        # 1. 构建配置
        self.config = TinyuConfig(
            hidden_size=req.hidden_size,
            num_hidden_layers=req.num_hidden_layers,
            num_attention_heads=req.num_attention_heads,
            num_key_value_heads=req.num_key_value_heads,
            use_moe=req.use_moe,
        )

        # 2. 初始化模型
        self.model, self.tokenizer = init_model(self.config, tokenizer_path="./model", device=self.device)
        print_model_param_details(self.model, detail=False)

        # 3. 加载预训练权重
        if not os.path.exists(req.weight_path):
            raise FileNotFoundError(f"权重文件不存在: {req.weight_path}")

        state_dict = torch.load(req.weight_path, map_location=self.device)

        # 兼容 DDP 保存的权重（去掉 module. 前缀）
        clean_state_dict = {}
        for k, v in state_dict.items():
            # 也可能权重本身没有包装在 checkpoint dict 里
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            clean_state_dict[new_key] = v

        # 如果保存的是完整 checkpoint 字典（包含 epoch、model_state_dict 等），需要提取
        if "model_state_dict" in clean_state_dict:
            clean_state_dict = clean_state_dict["model_state_dict"]

        # 严格加载基座权重
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.weight_info["base_weight"] = req.weight_path

        # 4. 如有 LoRA 权重则加载
        if req.lora_path and os.path.exists(req.lora_path):
            lora_state = torch.load(req.lora_path, map_location=self.device)
            # 同样兼容 checkpoint 格式
            if "model_state_dict" in lora_state:
                lora_state = lora_state["model_state_dict"]
            self.model.load_state_dict(lora_state, strict=False)
            self.weight_info["lora_weight"] = req.lora_path

        self.model.eval()
        self.weight_info["loaded_at"] = datetime.now().isoformat()

        return {
            "device": str(self.device),
            "base_weight": self.weight_info["base_weight"],
            "lora_weight": self.weight_info["lora_weight"],
            "params": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M",
        }

    def unload_model(self):
        """卸载模型，释放显存。"""
        self.model = None
        self.tokenizer = None
        self.config = None
        self.weight_info = {
            "base_weight": None,
            "lora_weight": None,
            "loaded_at": None,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_model_loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 50,
                        top_p: float = 0.9) -> Iterator[str]:
        """
        流式生成对话回复，逐 token 返回字符串片段。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")

        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        # 文本转 ID
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = input_ids[0].tolist()

        # Prefill 阶段
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values

        # 采样第一个 token
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token)

        # 解码并 yield
        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
        yield token_str

        if next_token == tokenizer.eos_token_id:
            return

        # 自回归生成
        for _ in range(max_new_tokens - 1):
            current_input = torch.tensor([[next_token]]).to(device)
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated_ids.append(next_token)
            token_str = tokenizer.decode([next_token], skip_special_tokens=True)
            yield token_str

            if next_token == tokenizer.eos_token_id:
                break

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """对 logits 进行 Top-K 和 Top-P 过滤。"""
        assert logits.dim() == 1
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits

    def get_status(self) -> dict:
        """获取当前引擎状态。"""
        return {
            "model_loaded": self.is_model_loaded(),
            "device": str(self.device) if self.device else None,
            **self.weight_info,
        }
