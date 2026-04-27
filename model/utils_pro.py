import torch
import torch.nn as nn
from typing import Optional


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 预计算频率并注册为 buffer (不会被优化器更新)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x shape: [B, num_heads, seq_len, head_dim]
        # position_ids shape: [B, seq_len]
        
        # 强制在 fp32 下计算位置编码
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # 计算角度: [B, head_dim//2, seq_len]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        # 拼接成 [B, seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 转换形状为 [B, 1, seq_len, head_dim] 以便完美广播
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, num_heads, seq_len, head_dim]
    # cos, sin: [B, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """处理 Grouped-Query Attention (GQA) 的 KV 复制"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)