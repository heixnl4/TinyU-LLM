import torch
from typing import Optional

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    # 优化：torch.arange(0, dim, 2) 长度天然为 dim // 2，直接移除冗余的切片操作
    freqs = 1 / rope_base ** (torch.arange(0, dim, 2).float() / dim)
    attn_factor = 1.0

    # 预留给 RoPE Scaling (如 NTK-Aware, Yarn 等) 的扩展接口
    # if rope_scaling is not None:
    #     orig_max, factor, beta_fast, beta_slow, attn_factor = (
    #     )

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    
    # 将频率拼接，适配旋转半个维度的操作
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    def rotate_half(x): 
        # 将特征维度的前半部分和后半部分进行互换并对前半部分取负
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(q.dtype)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b_size, seq_len, num_key_value_heads, head_dim = x.shape
    return (x[:, :, :, None, :]
            .expand(b_size, seq_len, num_key_value_heads, n_rep, head_dim)
            .reshape(b_size, seq_len, num_key_value_heads * n_rep, head_dim))
