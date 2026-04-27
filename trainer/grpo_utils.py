import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from trainer.train_utils import init_model  # 复用你预训练代码里的初始化函数
from trainer.rl_base import (
    TinyuValueModel,
    load_weights_safely,
    gather_logprobs,
    extract_rm_scores,
)


# =====================================================================
# 1. 核心初始化函数 (GRPO 专用：3 个模型，移除了 Critic)
# =====================================================================
def init_grpo_models(config, model_paths, device):
    """
    一次性初始化 GRPO 所需的 3 个模型 (移除了 Critic)
    """
    sft_path = model_paths.get("actor_sft", None)
    rm_path = model_paths.get("reward", None)

    # GRPO 中只有 Reward 的缺失参数（如 value_head）属于正常现象，不报警告
    ignore_missing = ["Reward"]

    # ---------------- 1. 初始化 Actor 和 Ref (生成架构) ----------------
    print(">>> 正在初始化 Actor 模型...")
    actor, tokenizer = init_model(config, device=device)
    load_weights_safely(actor, sft_path, device, "Actor", ignore_missing)

    print(">>> 正在初始化 Reference 模型...")
    ref_model, _ = init_model(config, device=device)
    load_weights_safely(ref_model, sft_path, device, "Reference", ignore_missing)

    # ---------------- 2. 初始化 RM (价值架构) ----------------
    print(">>> 正在初始化 Reward 模型...")
    reward_model = TinyuValueModel(config).to(device)
    load_weights_safely(reward_model, rm_path, device, "Reward", ignore_missing)

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


# =====================================================================
# 2. 经验收集 (GRPO 专用：含 group_size 的 repeat_interleave)
# =====================================================================
@torch.no_grad()  # 经验收集阶段绝对不能有梯度，否则必定 OOM
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
        rm_scores = extract_rm_scores(rm_outputs, full_attention_mask)

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


# =====================================================================
# 3. 优势估计 (GRPO 专用：组内归一化)
# =====================================================================
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
