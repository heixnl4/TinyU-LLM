"""
RL (PPO / GRPO) 公共基础模块
抽取 PPO 与 GRPO 共用的基础设施，避免代码重复。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torch.nn as nn
from model.model_Tinyu import TinyuForcausalLM
from trainer.train_utils import init_model


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
def load_weights_safely(model, ckpt_path, device, model_name="Model", ignore_missing=None):
    """安全地加载权重，处理分布式保存时带有的 module. 前缀"""
    if not ckpt_path:
        print(f"[{model_name}] 未提供权重路径，将使用随机初始化 (不推荐)。")
        return

    # 【新增逻辑】检查文件在磁盘上是否真实存在
    if not os.path.exists(ckpt_path):
        # 建议直接抛出异常而不是静默使用随机权重
        # raise FileNotFoundError(f"[{model_name}] 严重错误：权重文件路径 '{ckpt_path}' 不存在！请检查路径是否拼写正确。")

        # 如果你依然希望它哪怕找不到也强行随机初始化跑下去，可以换成下面这两行：
        print(f"[{model_name}] 严重警告：找不到权重文件 {ckpt_path}！将退化为随机初始化。")
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

    ignore_set = set(ignore_missing or [])
    if len(missing) > 0 and model_name not in ignore_set:
        # 只有非白名单内的模型缺失参数时才需要警惕
        print(f"[{model_name}] 警告：部分参数未加载: {missing[:5]}...")


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


# =====================================================================
# 3. Reward Model 输出适配（PPO / GRPO 共用）
# =====================================================================
def extract_rm_scores(rm_outputs, full_attention_mask):
    """
    鲁棒适配 Reward Model 输出，提取整句评分。

    支持两种输出形式：
      - 标量输出: 形状为 (batch_size,)
      - 序列输出: 形状为 (batch_size, seq_len)，自动取最后一个有效 token 的得分

    :param rm_outputs: Reward Model 原始输出
    :param full_attention_mask: 完整的 attention mask
    :return: rm_scores (batch_size,)
    """
    # 1. 剥离模型包裹类
    rm_tensor = rm_outputs.logits if hasattr(rm_outputs, "logits") else rm_outputs
    # 2. 挤掉末尾可能存在的 size 为 1 的冗余维度: (B, 1)->(B) 或是 (B, L, 1)->(B, L)
    rm_tensor = rm_tensor.squeeze(-1)

    if rm_tensor.dim() == 1:
        # 情况 A：模型已经是直接输出标量分了，形状为 (batch_size,)
        rm_scores = rm_tensor
    elif rm_tensor.dim() == 2:
        # 情况 B：模型输出的是序列稠密分，形状为 (batch_size, seq_len)
        # 修复：获取最后一个有效 token 的真实绝对索引
        # 构造一个形状相同的递增索引矩阵: [0, 1, 2, ..., seq_len-1]
        seq_length = full_attention_mask.shape[1]
        position_ids = torch.arange(seq_length, device=full_attention_mask.device)

        # 将 mask 为 0 的位置的索引清零，然后取每行的最大值即为最后一个 1 的索引
        last_valid_indices = (full_attention_mask * position_ids).argmax(dim=-1)

        # 利用 gather 从序列中精准抽出那个最末尾词的得分
        rm_scores = rm_tensor.gather(dim=1, index=last_valid_indices.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f"未知的 Reward Model 输出维度: {rm_tensor.shape}，请检查模型骨干。")

    return rm_scores
