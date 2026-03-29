from transformers import PretrainedConfig

class TinyuConfig(PretrainedConfig): 
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)

        self.dropout = kwargs.get("dropout", 0.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.flash_attn = kwargs.get("flash_attn", True)

        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

import torch
import torch.nn.functional as F
import math
from torch import nn
from typing import Optional


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return (self._norm(x.float()) * self.weight).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1 / rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (

        )

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): 
        return torch.cat((-x[..., x.shape(-1) // 2:], x[..., : x.shape(-1) // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b_size, seq_len, num_key_value_heads, head_dim = x.shape
    return (x[:, :, :, None, :]
            .expand(b_size, seq_len, num_key_value_heads, n_rep, head_dim)
            .reshape(b_size, seq_len, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    def __init__(self, config: TinyuConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        # 用户想禁用 GQA，用标准 MHA
        self.n_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size * self.head_dim)
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):

        b_size, seq_len, _ = x.shape
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        xq = xq.view(b_size, seq_len, self.n_heads, self.head_dim) 
        xk = xk.view(b_size, seq_len, self.n_kv_heads, self.head_dim) 
        xv = xv.view(b_size, seq_len, self.n_kv_heads, self.head_dim) 

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat(xk, past_key_value[0], dim=1)
            xv = torch.cat(xv, past_key_value[1], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 会调用的场景
            # 场景 1：模型训练（Training）
            # 场景 2：推理时的 Prefill 阶段（Prompt Processing）
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 方差归一化
            scores = (xq @ xk.transpose(-1, -2)) / math.sqrt(self.head_dim)

            # 因果掩码
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            # 填充掩码
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsequeeze(1).unsequeeze(2)) * -1e9
                scores += extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        
        output = output.transpose(1, 2).reshape(b_size, seq_len, -1)
        output = self.o_proj(output)
        output = self.resid_dropout(output)

        return output, past_kv



