import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torch.nn as nn
from model.model_Tinyu import TinyuModel, TinyuForcausalLM
from trainer.train_utils import init_model # 复用你预训练代码里的初始化函数

# =====================================================================
# 1. 价值模型包装器 (Value Model Wrapper) - 仅供 Reward Model 使用
# =====================================================================
class TinyuValueModel(TinyuForcausalLM):
    """
    专门为 Reward Model 设计的包装类 (GRPO 中不再需要 Critic)。
    在原本的 TinyU 骨干网络上，摘掉或忽略预测词表的 lm_head，
    换成一个输出维度为 1 的 value_head，用于给最后生成的完整句子打分。
    """
    def __init__(self, config):
        super().__init__(config)
        if not self.config.tie_word_embeddings and hasattr(self, "lm_head"):
            del self.lm_head
        # 核心：增加一个价值预测头，把 hidden_size 降维到 1
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 通过 Value Head 得到每个 token 的价值分数 shape: (batch_size, seq_len, 1)
        values = self.value_head(outputs[0])
        return values

# =====================================================================
# 2. 核心初始化函数
# =====================================================================
def load_weights_safely(model, ckpt_path, device, model_name="Model"):
    """安全地加载权重，处理分布式保存时带有的 module. 前缀"""
    if not ckpt_path:
        print(f"[{model_name}] 未提供权重路径，将使用随机初始化 (不推荐)。")
        return
    
    if not os.path.exists(ckpt_path):
        print(f"[{model_name}] 严重警告：找不到权重文件 {ckpt_path}！将退化为随机初始化。")
        return

    print(f"[{model_name}] 正在从 {ckpt_path} 加载权重...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[new_key] = v
        
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    
    if len(missing) > 0 and model_name not in ["Reward"]:
        print(f"[{model_name}] 警告：部分参数未加载: {missing[:5]}...")

def init_grpo_models(config, model_paths, device):
    """
    一次性初始化 GRPO 所需的 3 个模型 (移除了 Critic)
    """
    sft_path = model_paths.get("actor_sft", None)
    rm_path = model_paths.get("reward", None)

    # ---------------- 1. 初始化 Actor 和 Ref (生成架构) ----------------
    print(">>> 正在初始化 Actor 模型...")
    actor, tokenizer = init_model(config, device=device)
    load_weights_safely(actor, sft_path, device, "Actor")

    print(">>> 正在初始化 Reference 模型...")
    ref_model, _ = init_model(config, device=device)
    load_weights_safely(ref_model, sft_path, device, "Reference")

    # ---------------- 2. 初始化 RM (价值架构) ----------------
    print(">>> 正在初始化 Reward 模型...")
    reward_model = TinyuValueModel(config).to(device)
    load_weights_safely(reward_model, rm_path, device, "Reward")

    # ---------------- 3. 设置模型状态 ----------------
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
        
    actor.train()

    print("=== GRPO 三大模型初始化完毕 ===")
    return actor, ref_model, reward_model, tokenizer

def gather_logprobs(logits, labels):
    """
    从 Logits 中提取对应目标 Token 的对数概率 (Logprobs)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    gathered_logprobs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    return gathered_logprobs

@torch.no_grad() # 经验收集阶段绝对不能有梯度，否则必定 OOM
def generate_grpo_experience(actor, ref_model, reward_model, prompt_batch, tokenizer, device, ptdtype, group_size=4, kl_coef=0.1, max_response_length=256):
    """
    利用 Actor 对每个 Prompt 生成 G 个回复，并获取 Ref 和 Reward 模型的评估
    """
    # 1. 解析 Prompt 批次并将其拓展以匹配 Group Size
    prompt_input_ids = prompt_batch["input_ids"].to(device)
    prompt_attention_mask = prompt_batch["attention_mask"].to(device)
    prompt_len = prompt_input_ids.shape[1]

    # 【GRPO 核心改动】复制 Prompt，使得 batch_size 变为 batch_size * group_size
    prompt_input_ids = prompt_input_ids.repeat_interleave(group_size, dim=0)
    prompt_attention_mask = prompt_attention_mask.repeat_interleave(group_size, dim=0)

    # 2. Actor 生成回复 (Rollout)
    unwrapped_actor = actor.module if hasattr(actor, "module") else actor
    
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        # 使用 do_sample=True 且通常可提高 temperature 保证同一个 prompt 的 G 个回复具备多样性
        output_ids = unwrapped_actor.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_response_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.0, 
            top_p=0.95,
        )
    
    # 3. 切分出模型自己生成的 Response 部分
    responses = output_ids[:, prompt_len:]
    response_attention_mask = (responses != tokenizer.pad_token_id).long()
    
    full_input_ids = output_ids
    full_attention_mask = (full_input_ids != tokenizer.pad_token_id).long()

    # 4. 获取 Actor, Ref, Reward 三个模型的输出
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        actor_logits = actor(full_input_ids, attention_mask=full_attention_mask).logits
        ref_logits = ref_model(full_input_ids, attention_mask=full_attention_mask).logits
        rm_outputs = reward_model(full_input_ids, attention_mask=full_attention_mask)

        # ========== 鲁棒适配 RM 输出 ==========
        rm_tensor = rm_outputs.logits if hasattr(rm_outputs, "logits") else rm_outputs
        rm_tensor = rm_tensor.squeeze(-1) 
        
        if rm_tensor.dim() == 1:
            rm_scores = rm_tensor
        elif rm_tensor.dim() == 2:
            seq_length = full_attention_mask.shape[1]
            position_ids = torch.arange(seq_length, device=full_attention_mask.device)
            last_valid_indices = (full_attention_mask * position_ids).argmax(dim=-1)
            rm_scores = rm_tensor.gather(dim=1, index=last_valid_indices.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"未知的 Reward Model 输出维度: {rm_tensor.shape}，请检查模型骨干。")
    
    # 5. 对齐 Logits 序列以计算 Logprobs
    actor_logits = actor_logits[:, prompt_len - 1 : -1, :]
    ref_logits = ref_logits[:, prompt_len - 1 : -1, :]
    
    actor_logprobs = gather_logprobs(actor_logits, responses)
    ref_logprobs = gather_logprobs(ref_logits, responses)
    
    # 6. 计算 Token 级别的 KL 散度惩罚
    kl_penalties = -kl_coef * (actor_logprobs - ref_logprobs)
            
    # 7. Mask 掩盖无效 Pad 部分
    actor_logprobs = actor_logprobs * response_attention_mask
    ref_logprobs = ref_logprobs * response_attention_mask
    kl_penalties = kl_penalties * response_attention_mask

    # 封装返回。注: 为了兼容主脚本习惯，我们将整句评分返回在 'rewards' 键位。
    return {
        "input_ids": full_input_ids,            
        "attention_mask": full_attention_mask,
        "prompts": prompt_input_ids,
        "actions": responses,                    
        "logprobs": actor_logprobs,              
        "rewards": rm_scores,             # (B * G) 级别的环境整句得分
        "kl_penalties": kl_penalties,     # (B * G, SeqLen) 级别的 KL 惩罚
        "response_mask": response_attention_mask
    }

@torch.no_grad()
def compute_grpo_advantages(rm_scores: torch.Tensor, kl_penalties: torch.Tensor, response_mask: torch.Tensor, group_size: int):
    """
    计算 GRPO 的组内优势 (Group Relative Advantages)
    
    :param rm_scores: Reward模型输出的整句评分，形状为 (batch_size * group_size)
    :param kl_penalties: Token级别的KL惩罚项，形状为 (batch_size * group_size, seq_len)
    :param response_mask: 掩码矩阵，形状为 (batch_size * group_size, seq_len)
    :param group_size: GRPO 组大小 (通常为 4~8)
    :return: 归一化且包含 KL 的优势矩阵，形状与 kl_penalties 相同
    """
    total_size = rm_scores.shape[0]
    batch_size = total_size // group_size
    
    # ================= 1. 计算组内标准化的全局 Advantage =================
    # 将 (B*G) 重塑为 (B, G) 以在组内进行计算
    rm_scores_reshaped = rm_scores.view(batch_size, group_size)
    
    # 计算均值和标准差 (带上 keepdim 方便直接广播减法)
    mean = rm_scores_reshaped.mean(dim=1, keepdim=True)
    std = rm_scores_reshaped.std(dim=1, keepdim=True)
    
    # 组内归一化 (Z-Score)
    group_advantages = (rm_scores_reshaped - mean) / (std + 1e-8)
    
    # 拉回一维形状 (B * G)
    group_advantages = group_advantages.view(-1)
    
    # ================= 2. 计算最终 Token 级优势 =================
    # GRPO 的 Advantage 是: 全局归一化得分 + Token级别 KL 惩罚
    # 将 (B*G) 拓展为 (B*G, 1) 从而能够与 (B*G, SeqLen) 的 kl_penalties 相加
    token_advantages = group_advantages.unsqueeze(1) + kl_penalties
    
    # 施加 Mask 防止 padding 位置影响 Loss
    token_advantages = token_advantages * response_mask
    
    return token_advantages