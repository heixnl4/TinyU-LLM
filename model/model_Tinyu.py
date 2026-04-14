import torch
import torch.nn.functional as F
import math
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from .configuration import TinyuConfig
from .utils import precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return (self._norm(x.float()) * self.weight).type_as(x)


class Attention(nn.Module):
    def __init__(self, config: TinyuConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        # 如果禁用 GQA，则用标准 MHA
        self.n_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.dropout = config.dropout
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
            xk = torch.cat(past_key_value[0], xk, dim=1)
            xv = torch.cat(past_key_value[1], xv, dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 会调用的场景
            # 场景 1：模型训练（Training）
            # 场景 2：推理时的 Prefill 阶段（Prompt Processing）
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            # 方差归一化
            scores = (xq @ xk.transpose(-1, -2)) / math.sqrt(self.head_dim)

            # 因果掩码
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            # 填充掩码
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                scores += extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        
        output = output.transpose(1, 2).reshape(b_size, seq_len, -1)
        output = self.o_proj(output)
        output = self.resid_dropout(output)

        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: TinyuConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))


class MOEFeedForward(nn.Module):
    def __init__(self, config: TinyuConfig):
        super().__init__()
        self.config = config

        # 共享专家
        if config.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config, intermediate_size=config.moe_intermediate_size) 
                for _ in range(config.num_shared_experts)
            ])

        # 路由专家
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size) 
            for _ in range(config.num_experts)])

        # 路由器
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False) 

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # 计算共享专家的输出
        shared_output = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(x_flat)

        # 计算路由专家的输出
        gate = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(gate, k=self.config.num_experts_per_token, dim=-1, sorted=False)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        routed_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            chosen_experts = (topk_idx == i)
            if chosen_experts.any():
                token_idx = chosen_experts.any(dim=-1).nonzero().squeeze()
                weigth = topk_weight[chosen_experts].view(-1, 1)
                routed_output.index_add_(dim=0, index=token_idx, source=(weigth * expert(x_flat[token_idx])).to(routed_output.dtype))
            elif self.training:
            # 只在训练时特殊处理没使用的experts 
                routed_output[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        
        # 负载均衡辅助损失
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)     # (K, E)
            self.aux_loss = (load * gate.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef    # gate.mean(0)  [E,]
        else:
            self.aux_loss = gate.new_zeros(1).squeeze()

        # 最终输出合并
        final_output = shared_output + routed_output
        return final_output.view(batch_size, seq_len, hidden_size)


class TinyuBlock(nn.Module):
    def __init__(self, layer_id: int, config: TinyuConfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, past_key_value, use_cache=False, attention_mask=None):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.attn(
            hidden_states, 
            position_embeddings, 
            past_key_value, 
            use_cache, 
            attention_mask)
        hidden_states += residual

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(hidden_states)

        return hidden_states, present_key_value
    

class TinyuModel(nn.Module):
    def __init__(self, config: TinyuConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TinyuBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False, attention_mask=None):
        b_size, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layer'):
            past_key_values = None
        
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos : start_pos + seq_len], self.freqs_sin[start_pos : start_pos + seq_len])

        presents = []

        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, 
                position_embeddings, 
                past_key_value, 
                use_cache, 
                attention_mask)
            
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss
    

class TinyuForcausalLM(PreTrainedModel, GenerationMixin):
    # 声明该模型类对应的配置类,是 Hugging Face PreTrainedModel 子类的标准要求
    config_class = TinyuConfig

    def __init__(self, config: TinyuConfig = None):
        self.config = config or TinyuConfig()
        super().__init__(self.config)
        self.model = TinyuModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            # 如果配置为 True，则输入 Embedding 和输出 Head 共享参数（省显存，但表达能力受限）
            self.model.embed_tokens.weight = self.lm_head.weight


    def forward(self, input_ids, past_key_values=None, use_cache=False, attention_mask=None, logits_to_keep=0, labels=None):
        hidden_states, presents, aux_loss = self.model(input_ids, past_key_values, use_cache, attention_mask)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)


