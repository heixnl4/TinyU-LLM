import torch
from typing import Optional

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    freqs = 1 / rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    attn_factor = 1.0

    # if rope_scaling is not None:
    #     orig_max, factor, beta_fast, beta_slow, attn_factor = (

    #     )

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): 
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
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
