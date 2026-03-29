from transformers import PretrainedConfig

class TinyuConfig(PretrainedConfig): 
    def __init__(
            self, 
            hidden_size: int = 512,     
            num_hidden_layers: int = 8,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 7000,
            max_position_embeddings: int = 32768,

            dropout: float = 0.0,
            rms_norm_eps: float = 1e-5,
            rope_theta: float = 1000000.0,

            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = "silu",
            inference_rope_scaling: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.inference_rope_scaling = inference_rope_scaling

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
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.vocab_size = config.vocab_size
        