import torch
# 假设你的配置类和模型类在 modeling_tinyu.py 中
# from modeling_tinyu import TinyuConfig, TinyuForcausalLM
from configuration import TinyuConfig # 请根据实际路径调整
from model_Tinyu import TinyuForcausalLM # 请根据实际路径调整

def test_generation():
    print("=== 初始化测试配置 ===")
    # 1. 伪造一个极简的 Config 用于测试
    config = TinyuConfig(
        vocab_size=6400,           # 词表大小
        hidden_size=256,            # 隐藏层维度
        num_hidden_layers=2,        # 层数（测试给2层即可）
        num_attention_heads=8,      # 注意力头数
        num_key_value_heads=4,      # KV 头数 (GQA)
        head_dim=32,                # 每个头的维度 (hidden_size / num_attention_heads)
        max_position_embeddings=512,# 最大上下文长度
        use_moe=True,               # 开启 MoE
        num_experts=4,              # 路由专家数量
        num_experts_per_token=2,    # Top-K 专家
        num_shared_experts=1,       # 共享专家数量
        moe_intermediate_size=512,  # 专家中间层维度
        flash_attn=False            # 本地 CPU 测试时关闭 Flash Attention
    )

    # 2. 实例化模型并切换到 eval 模式
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载模型到设备: {device}...")
    model = TinyuForcausalLM(config).to(device)
    model.eval()

    # 3. 构造 Dummy Input (模拟一段长度为 5 的 Prompt)
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    print(f"输入 Prompt 的形状: {input_ids.shape}")
    print(f"输入 Prompt Token IDs: {input_ids.tolist()}")

    # 4. 调用 generate 方法
    print("\n=== 开始生成 ===")
    max_new_tokens = 20
    
    with torch.no_grad(): # 推理时一定要关闭梯度
        generated_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,             # 开启采样
            temperature=0.7,            # 控制生成多样性
            top_k=50,                   # Top-K 采样
            use_cache=True,             # 开启 KV Cache (核心测试点)
            pad_token_id=0,             # 随便指定一个 padding token
            eos_token_id=config.vocab_size - 1, # 随便指定一个 eos token
            return_dict_in_generate=True, # 方便查看额外信息
            output_scores=True          # 获取生成的 logits
        )

    # 5. 打印结果
    generated_sequences = generated_output.sequences
    print(f"\n生成完毕！总序列形状: {generated_sequences.shape}")
    print(f"最终输出的 Token IDs: {generated_sequences.tolist()}")
    
    # 验证生成的长度是否符合预期
    expected_length = seq_len + max_new_tokens
    actual_length = generated_sequences.shape[1]
    
    if actual_length > seq_len:
        print("✅ Generate 测试通过！KV Cache 运行正常。")
    else:
        print("❌ 生成失败，未生成新 Token。")

if __name__ == "__main__":
    test_generation()