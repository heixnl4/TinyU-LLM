import os
import torch
import torch.nn.functional as F
from transformers import TextStreamer # 或者使用你自定义的 Tokenizer 导入方式
from model.configuration import TinyuConfig
from trainer.train_utils import init_model
from trainer.arguments import inference_args

# ================= 1. 核心采样算法 =================
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    对模型的原始输出分数 (Logits) 进行 Top-K 和 Top-P (Nucleus) 截断，
    这是防止大模型胡言乱语、提升生成质量的绝对核心逻辑。
    """
    assert logits.dim() == 1  # 确保输入是单维度张量
    
    top_k = min(top_k, logits.size(-1))  # 安全检查
    if top_k > 0:
        # 移除非 Top-K 的 token 概率
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # 对 logits 排序并计算累积概率
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 把阈值右移一位，确保留下第一个超过 top_p 的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits

# ================= 2. 自回归生成逻辑 =================
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, device="cpu"):
    """
    手写的自回归生成循环，直观展示大模型是如何“一个词一个词”蹦出来的。
    """
    model.eval() # 切换到评估模式，关闭 Dropout 等随机性
    
    # 1. 文本转 ID
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    print(f"\n[Prompt]: {prompt}")
    print("[Model]: ", end="", flush=True)
    
    generated_ids = input_ids[0].tolist()

    # 🌟 新增：初始化 KV Cache 变量
    past_key_values = None

    # 🌟 第一阶段：Prefill (预填充阶段)
    # 把用户输入的整段 Prompt 一次性送入模型，建立初始的 KV Cache
    # 注意：此时模型的 forward 需要支持 use_cache=True 并返回 past_key_values
    outputs = model(input_ids, use_cache=True)
    next_token_logits = outputs.logits[0, -1, :]
    past_key_values = outputs.past_key_values  # 拿到第一批缓存

    # 采样算出第一个新词
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    generated_ids.append(next_token)
    print(tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

    if next_token == tokenizer.eos_token_id:
        print("\n" + "-"*50)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 2. 核心自回归循环
    for _ in range(max_new_tokens - 1):
        # 🌟 核心性能起飞点：当前的 input 永远只有 1 个 token (即上一步刚生成的词)！
        # 不再是把整个 generated_ids 塞进去了！
        current_input = torch.tensor([[next_token]]).to(device)
        
        # 🌟 前向传播：传入 current_input 并且传入 past_key_values
        outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = outputs.logits[0, -1, :] 
        past_key_values = outputs.past_key_values
        
        # 应用 Temperature 缩放
        # 公式: q_i = exp(z_i / T) / sum(exp(z_j / T))
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        # 应用 Top-K 和 Top-P 过滤
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        
        # 转化为概率分布并采样
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # 把新预测出的 token 加到序列尾部，进入下一轮循环
        generated_ids.append(next_token)
        
        # 实时流式打印 (流式输出机制)
        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
        print(token_str, end="", flush=True)
        
        # 如果生成了 EOS (结束符)，提前终止
        if next_token == tokenizer.eos_token_id:
            break
            
    print("\n" + "-"*50)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ================= 3. 模型加载与启动 =================
def main():
    args = inference_args()

    # 智能设备选择：支持 CUDA，并兼容 Apple Silicon (MPS) 本地开发，最后回退到 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"当前推理使用设备: {device}")
    
    # 权重路径和分词器路径
    arch_signature = f"h{args.hidden_size}_l{args.num_hidden_layers}_ah{args.num_attention_heads}_moe{int(args.use_moe)}"
    output_dir = os.path.join(args.output_dir, f"{args.run_name}_{arch_signature}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/pretrain_checkpoint.pth" 
    tokenizer_path = "./model" 
    
    # 2. 初始化模型架构 (需要和预训练时保持绝对一致)
    config = TinyuConfig(
        hidden_size=512, 
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    model, tokenizer = init_model(config, tokenizer_path, device=device)

    
    # 3. 剥离 DDP 的 `module.` 前缀并加载权重，单卡训练不需要
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[new_key] = v
        
    model.load_state_dict(clean_state_dict)
    print("模型权重加载成功！\n")
    
    # 4. 开启对话/续写测试
    prompts = [
        "人工智能的未来发展趋势是",
        "你是谁",
        "写一下快速排序的代码"
    ]

    for p in prompts:
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            max_new_tokens=150,
            temperature=0.7, # 温度越低越严谨，越高越发散
            top_k=50,
            top_p=0.9,
            device=device
        )

if __name__ == "__main__":
    main()