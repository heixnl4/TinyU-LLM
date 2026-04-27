import torch
import torch.nn.functional as F
from trainer.train_utils import init_model # 复用你预训练代码里的初始化函数
from trainer.rl_base import load_weights_safely


def init_dpo_models(config, actor_model_path, device):
    """
    一次性初始化 DPO 所需的 2 个模型 (Actor 和 Ref)
    """
    print(">>> 正在初始化 Actor 模型...")
    actor, tokenizer = init_model(config, device=device)
    load_weights_safely(actor, actor_model_path, device, "Actor")

    print(">>> 正在初始化 Reference 模型...")
    ref_model, _ = init_model(config, device=device)
    load_weights_safely(ref_model, actor_model_path, device, "Reference")

    # 冻结 Reference 模型
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    actor.train()
    print("=== DPO 模型初始化完毕 ===")
    return actor, ref_model, tokenizer

def get_batch_logprobs(logits, labels, attention_mask, prompt_len):
    """
    获取一个 batch 内生成 Response 部分的 logprobs 总和。
    注意：DPO 计算的是整句话的 logprobs 之和，而不是每个 token 独立算。
    """
    # 截取 logits 去匹配预测下一个词 (labels的偏移)
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    attention_mask = attention_mask[:, 1:]

    # 提取所有位置的 logprobs
    log_probs = F.log_softmax(logits, dim=-1)
    gathered_logprobs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # 制作 mask：我们要忽略 pad token，并且【忽略 prompt 部分】
    # 生成一个与 labels 同样长度的递增索引矩阵
    seq_length = labels.shape[1]
    position_ids = torch.arange(seq_length, device=labels.device).unsqueeze(0).expand_as(labels)
    
    # 真正的 loss mask：既不是 padding，索引又大于等于 prompt_len - 1 (因为整体向左移了一位)
    loss_mask = (attention_mask == 1) & (position_ids >= (prompt_len - 1))

    # 应用 mask 并沿着序列长度求和，得到整句话的概率
    gathered_logprobs = gathered_logprobs * loss_mask
    return gathered_logprobs.sum(dim=-1)

def compute_dpo_loss(policy_chosen_logprobs, policy_rejected_logprobs, 
                     reference_chosen_logprobs, reference_rejected_logprobs, beta):
    """
    计算 DPO Loss
    """
    # 1. 计算当前 Policy 模型对好坏回答的倾向性差距
    pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    
    # 2. 计算 Reference 模型的固有基准差距
    ref_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    
    # 3. 计算对数几率差 (Logits for the sigmoid)
    logits = pi_logratios - ref_logratios
    
    # 4. 经过 Sigmoid 并取对数，作为最终的分类 Loss
    # 数学上等价于: -F.logsigmoid(beta * logits)
    losses = -F.logsigmoid(beta * logits)
    
    # 返回平均 loss 以及所选择的奖励幅度 (用于日志监控)
    chosen_rewards = beta * (policy_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = beta * (policy_rejected_logprobs - reference_rejected_logprobs).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
    
    return losses.mean(), reward_accuracies