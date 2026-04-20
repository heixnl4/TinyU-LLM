import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from model.model_Tinyu import TinyuModel, TinyuForcausalLM
from trainer.train_utils import init_model # 复用你预训练代码里的初始化函数

# =====================================================================
# 1. 价值模型包装器 (Value Model Wrapper)
# =====================================================================
class TinyuValueModel(TinyuForcausalLM):
    """
    专门为 Critic 和 Reward Model 设计的包装类。
    在原本的 TinyU 骨干网络上，摘掉或忽略预测词表的 lm_head，
    换成一个输出维度为 1 的 value_head，用于给当前状态打分。
    """
    def __init__(self, config):
        super().__init__(config)
        # 父类初始化时顺带创建了预测词表的 self.lm_head，它在 Value 模型里完全没用。
        # 如果你的词表很大，它会白白吃掉极多的显存。
        # 注意：如果开启了权重共享 (tie_word_embeddings)，那就不能删，否则会连带着把输入层的 Embedding 也删掉。
        if not self.config.tie_word_embeddings and hasattr(self, "lm_head"):
            del self.lm_head
        # 核心：增加一个价值预测头，把 hidden_size 降维到 1
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # 1. 跑基础模型的前向传播，取出最后一层的隐藏状态 (hidden states)
        # 注意：这里需要你的 base_model 支持返回 hidden_states
        # 如果你的基础模型是标准的，可能类似于 output.hidden_states[-1]
        
        # 以下是一种通用兼容写法，具体请根据你 TinyU 模型的 forward 返回值调整：
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 2. 通过 Value Head 得到每个 token 的价值分数 shape: (batch_size, seq_len, 1)
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

    print(f"[{model_name}] 正在从 {ckpt_path} 加载权重...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # 清理 DDP 保存时可能引入的 "module." 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[new_key] = v
        
    # strict=False 允许加载时有一点点不匹配 (比如 Critic 的 value_head 在 SFT 权重里是没有的)
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    
    if len(missing) > 0 and model_name not in ["Critic", "Reward"]:
        # 只有 Actor/Ref 缺失参数时才需要警惕，Critic 缺失 value_head 是正常的
        print(f"[{model_name}] 警告：部分参数未加载: {missing[:5]}...")

def init_ppo_models(config, model_paths, device):
    """
    一次性初始化 PPO 所需的 4 个模型
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

    # ---------------- 2. 初始化 Critic 和 RM (价值架构) ----------------
    print(">>> 正在初始化 Critic 模型...")
    # 先用 init_model 创建骨架，再套上 ValueModel 的壳子
    critic = TinyuValueModel(config, device=device).to(device)
    # Critic 可以从 Reward Model 初始化（推荐），也可以从 SFT 初始化
    load_weights_safely(critic, rm_path or sft_path, device, "Critic")

    print(">>> 正在初始化 Reward 模型...")
    reward_model = TinyuValueModel(config, device=device).to(device)
    load_weights_safely(reward_model, rm_path, device, "Reward")

    # ---------------- 3. 设置模型状态 ----------------
    # 明确设置不需要梯度的模型，避免训练时误操作或浪费显存
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
        
    actor.train()
    critic.train()

    print("=== PPO 四大模型初始化完毕 ===")
    return actor, critic, ref_model, reward_model, tokenizer

def gather_logprobs(logits, labels):
    """
    从 Logits 中提取对应目标 Token 的对数概率 (Logprobs)
    :param logits: 形状为 (batch_size, seq_len, vocab_size)
    :param labels: 形状为 (batch_size, seq_len)
    """
    # 沿着 vocab_size 维度计算 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    # 利用 gather 提取 labels 对应的位置的概率
    # 需要先将 labels 扩维到 (batch, seq, 1) 才能与 log_probs (batch, seq, vocab) 对齐
    gathered_logprobs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    return gathered_logprobs

@torch.no_grad() # 经验收集阶段绝对不能有梯度，否则必定 OOM
def generate_experience(actor, critic, ref_model, reward_model, prompt_batch, tokenizer, device, ptdtype, kl_coef=0.1, max_response_length=256):
    """
    利用 Actor 生成回复，并获取 Critic、Ref 和 Reward 模型的评估
    """
    # 1. 解析 Prompt 批次
    prompt_input_ids = prompt_batch["input_ids"].to(device)
    prompt_attention_mask = prompt_batch["attention_mask"].to(device)
    prompt_len = prompt_input_ids.shape[1]

    # 2. Actor 生成回复 (Rollout)
    # 如果 Actor 被 DDP 包装过，必须调用 .module 才能使用 HF 的 generate 方法
    unwrapped_actor = actor.module if hasattr(actor, "module") else actor
    
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        # 使用 do_sample=True 保证探索性
        output_ids = unwrapped_actor.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_response_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    
    # 3. 切分出模型自己生成的 Response 部分
    # output_ids 的结构是 [Prompt_Tokens, Response_Tokens]
    responses = output_ids[:, prompt_len:]
    # 生成 response 对应的 attention_mask
    response_attention_mask = (responses != tokenizer.pad_token_id).long()
    
    full_input_ids = output_ids
    full_attention_mask = (full_input_ids != tokenizer.pad_token_id).long()

    # 4. 获取 Actor, Ref, Critic, Reward 四个模型的输出
    with torch.autocast(device_type="cuda", dtype=ptdtype):
        # (1) Actor 的当前策略分布
        actor_logits = actor(full_input_ids, attention_mask=full_attention_mask).logits
        
        # (2) Reference 模型的参考分布
        ref_logits = ref_model(full_input_ids, attention_mask=full_attention_mask).logits
        
        # (3) Critic 预估的价值 (Value) -> 假设输出 shape (batch_size, seq_len, 1)
        values = critic(full_input_ids, attention_mask=full_attention_mask).squeeze(-1)
        
        # (4) Reward Model 打分 -> 假设输出一个标量分数 (batch_size,)
        # 如果你的 RM 输出是 sequence，需要自己提取末尾有效 token 的打分
        rm_outputs = reward_model(full_input_ids, attention_mask=full_attention_mask)
        # 此处做一个简单适配，假设 RM 直接吐出标量分或列表
        rm_scores = rm_outputs.logits.squeeze(-1) if hasattr(rm_outputs, "logits") else rm_outputs

    # 5. 对齐 Logits 序列：
    # 对于因果语言模型，第 t 个位置的输出 logits 是用来预测第 t+1 个 token 的。
    # 因此我们要预测 response_tokens，需要使用 prompt_len - 1 到 -1 的 logits。
    actor_logits = actor_logits[:, prompt_len - 1 : -1, :]
    ref_logits = ref_logits[:, prompt_len - 1 : -1, :]
    
    # 从分布中抽取出 Actor 生成的那些词的专属概率
    actor_logprobs = gather_logprobs(actor_logits, responses)
    ref_logprobs = gather_logprobs(ref_logits, responses)
    
    # 截取 Response 对应的 Value
    values = values[:, prompt_len - 1 : -1]

    # 6. 计算 KL 散度与最终步步级奖励 (Step-wise Rewards)
    # PPO 要求模型不偏离初始(SFT)模型太远，因此要引入 KL 惩罚
    kl_penalty = -kl_coef * (actor_logprobs - ref_logprobs)
    
    # 初始的每一步 reward 只有 KL 惩罚
    rewards = kl_penalty.clone()
    
    # 遍历每个 batch 将 Reward Model 给出的最终得分加在句末
    for i in range(responses.shape[0]):
        # 找到这局对话生成的有效长度（排掉末尾的 Pad）
        valid_len = response_attention_mask[i].sum().item()
        if valid_len > 0:
            end_idx = valid_len - 1
            # 将环境奖励（RM 分数）施加在生成的最后一个词上
            rewards[i, end_idx] += rm_scores[i]
            
    # 7. Mask 掩盖无效 Pad 部分 (防止把 Padding 算入 Loss 导致模型崩溃)
    actor_logprobs = actor_logprobs * response_attention_mask
    ref_logprobs = ref_logprobs * response_attention_mask
    values = values * response_attention_mask
    rewards = rewards * response_attention_mask

    # 封装并返回所有经验数据
    return {
        "input_ids": full_input_ids,             # 完整的 tokens，供之后计算更新 loss 用
        "attention_mask": full_attention_mask,
        "prompts": prompt_input_ids,
        "actions": responses,                    # Actor 实际选择的词
        "logprobs": actor_logprobs,              # Actor 选择这些词的旧概率 (old_logprobs)
        "values": values,                        # Critic 给出的估值
        "rewards": rewards                       # 融合了 KL 的最终奖励序列
    }


@torch.no_grad()
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    """
    计算广义优势估计 (GAE) 和 回报 (Returns)
    
    :param rewards: 每一步的真实/KL混合奖励，形状为 (batch_size, seq_len)
    :param values: Critic模型预测的每一步价值，形状为 (batch_size, seq_len)
    :param gamma: 折扣因子 (Discount factor)，决定对未来奖励的看重程度
    :param lam: GAE 平滑参数 (Lambda)，用于权衡偏差和方差
    :return: advantages (优势), returns (回报)
    """
    # 确保没有多余的梯度流
    batch_size, seq_len = rewards.shape
    
    # 初始化 advantages 容器，保持与 rewards 相同的设备和数据类型
    advantages = torch.zeros_like(rewards)
    
    # 记录上一步的 GAE 累加值
    last_gae_lam = 0.0
    
    # ================= 核心逻辑 =================
    # 从序列的最后一个 Token 开始，倒序往回算
    for t in reversed(range(seq_len)):
        # 如果是序列的最后一步，下一步的预测价值 (next_value) 默认视为 0
        if t == seq_len - 1:
            next_value = 0.0
        else:
            next_value = values[:, t + 1]
            
        # 1. 计算 TD Error (时序差分误差)
        # 公式: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # 这代表着：“真实的单步奖励 + 对未来的预期” 比 “你原本觉得现在有多好” 多出了多少
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        
        # 2. 计算并累加当前步的优势 (Advantage)
        # 公式: A_t = delta_t + gamma * lambda * A_{t+1}
        # 如果 lam = 0，A_t 就完全等于单步的 delta_t (方差低，偏差高)
        # 如果 lam = 1，A_t 就是经典的蒙特卡洛回报减去基线 (方差高，偏差低)
        advantages[:, t] = last_gae_lam = delta + gamma * lam * last_gae_lam

    # ================= 计算 Returns =================
    # 回报 (Returns) 是真实获得的价值，用来指导 Critic 网络更新自己
    # 数学上，Returns = Advantages + Values
    returns = advantages + values
    
    return advantages, returns