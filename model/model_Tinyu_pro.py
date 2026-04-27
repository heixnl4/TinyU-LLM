import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from .configuration import TinyuConfig
from .utils_pro import RotaryEmbedding, apply_rotary_pos_emb, repeat_kv


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # 强制在 fp32 下计算方差，防止混合精度训练时溢出
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, config: TinyuConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU() 

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    def __init__(self, config: TinyuConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        
        self.num_shared_experts = config.num_shared_experts
        self.shared_experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size) 
            for _ in range(self.num_shared_experts)
        ]) if self.num_shared_experts > 0 else None

        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size) 
            for _ in range(self.num_experts)
        ])
        
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # 1. Shared Experts Computation
        shared_output = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(x_flat)

        # 2. Router Computation
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # 重新归一化权重
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Routed Experts Computation
        routed_output = torch.zeros_like(x_flat)
        
        # 为 DDP 训练时的 dummy loss 准备
        dummy_losses = []
        
        for i, expert in enumerate(self.experts):
            # 获取选择当前专家的 token 掩码 (num_tokens, top_k)
            expert_match = (topk_indices == i)
            token_mask = expert_match.any(dim=-1)
            
            if token_mask.any():
                token_indices = token_mask.nonzero().flatten()
                expert_inputs = x_flat[token_indices]
                
                # 提取对应的归一化权重并重塑形状以便广播乘法
                weights_for_expert = topk_weights[expert_match].view(-1, 1)
                
                expert_outputs = expert(expert_inputs) * weights_for_expert.to(expert_inputs.dtype)
                routed_output.index_add_(0, token_indices, expert_outputs)
            elif self.training:
                # 确保所有专家都参与计算图，防止 DDP 报错
                dummy_losses.append(sum(p.sum() for p in expert.parameters()) * 0.0)
                
        if dummy_losses:
            routed_output[0] += sum(dummy_losses).to(routed_output.dtype)

        # 4. Auxiliary Loss Calculation
        if self.training and self.config.router_aux_loss_coef > 0:
            # 计算负载率 (Load) 和 平均门控概率 (Importance)
            tokens_per_expert = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=(0, 1))
            load = tokens_per_expert / x_flat.shape[0] 
            importance = gate_probs.mean(dim=0)
            
            self.aux_loss = (load * importance).sum() * self.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 5. Combine and Reshape
        final_output = shared_output + routed_output
        return final_output.view(batch_size, seq_len, hidden_dim)
    

# ==========================================
# 3. 核心注意力层 (Attention)
# ==========================================
class Attention(nn.Module):
    def __init__(self, config: TinyuConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        # 1. 投影并【及早转置】 -> [Batch, Heads, Seq_Len, Head_Dim]
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. 旋转位置编码 (RoPE)
        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. 缓存管理 (DynamicCache)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # 4. GQA 复制 (如果需要)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 5. 计算 Attention (优先使用原生 SDPA / FlashAttention)
        is_causal = True if attention_mask is None and q_len > 1 else False
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask, # 如果有自定义的 Padding Mask 则传入
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=is_causal
        )

        # 6. 转置回原样并合并 Heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 7. 输出投影
        attn_output = self.o_proj(attn_output)
        return attn_output


# ==========================================
# 4. Transformer Block 与 主干网络
# ==========================================
class DecoderLayer(nn.Module):
    def __init__(self, config: TinyuConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        # Pre-Norm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rotary_emb=rotary_emb,
        )
        hidden_states = residual + hidden_states

        # Pre-Norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class StandardModel(PreTrainedModel):
    config_class = TinyuConfig

    def __init__(self, config: TinyuConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, 
            max_position_embeddings=config.max_position_embeddings, 
            base=config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        batch_size, seq_length = input_ids.shape

        # 1. 初始化 Cache (如果处于生成模式且刚开始)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # 2. 处理 Position IDs (基于历史缓存长度计算当前序列的位置)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # 3. 处理 Attention Mask (仅当提供 padding 时需要，因果掩码由 SDPA 内部处理)
        if attention_mask is not None and attention_mask.dim() == 2:
            # 扩展 mask 形状为 [B, 1, Q_len, KV_len] 以适配 SDPA
            target_length = seq_length + (past_key_values.get_seq_length() if past_key_values is not None else 0)
            expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, target_length).to(torch.bool)
            attention_mask = ~expanded_mask # SDPA 要求需要忽略的位置为 True

        hidden_states = self.embed_tokens(input_ids)

        # 4. 依次经过所有 Transformer 层
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                rotary_emb=self.rotary_emb,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


# ==========================================
# 5. 顶层封装模型 (For Causal LM)
# ==========================================
class StandardForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyuConfig
    _no_split_modules = ["DecoderLayer"] # 帮助 Accelerate/DeepSpeed 分配层

    def __init__(self, config: TinyuConfig):
        super().__init__(config)
        self.model = StandardModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 走主干网络
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 计算 Logits
        logits = self.lm_head(hidden_states)

        # 计算 Loss
        loss = None
        if labels is not None:
            # 错位计算: 模型基于 token i 预测 token i+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # 必须实现：为生成流程（Generation）准备裁剪后的输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 如果缓存中存在历史记录，说明正处于自回归 Decode 阶段
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            
            # 如果提供了 attention_mask，必须用它推断出正确的长度（处理左侧 Padding 的情况）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 动态生成 Position IDs
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }