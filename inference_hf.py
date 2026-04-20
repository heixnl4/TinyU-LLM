import os
import torch
from transformers import AutoTokenizer, TextStreamer
from model.configuration import TinyuConfig
from trainer.train_utils import init_model
from trainer.arguments import inference_args

# ================= 1. 生成与流式输出逻辑 =================
@torch.no_grad()
def generate_text_hf(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, device="cpu"):
    """
    使用 Hugging Face 原生的 model.generate() 配合 TextStreamer 实现流式输出
    """
    model.eval()
    
    # 将输入转为 tensor 并移至对应设备
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, truncation=True).to(device)
    
    print(f"\n[Prompt]: {prompt}")
    print("[Model]: ", end="", flush=True)
    
    # 初始化流式输出器
    # skip_prompt=True 表示不在终端里重复打印你输入的 Prompt
    # skip_special_tokens=True 表示不打印类似 <|endoftext|> 这样的特殊符号
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 🌟 调用底层的 generate 方法
    # 注意：generate 会自动在底层处理循环、采样和 KV Cache
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,  # 必须开启采样，否则 top_k 和 top_p 不生效，退化为贪心搜索
        streamer=streamer, # 传入流式输出器
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    print("\n" + "-"*50)
    
    # 如果你需要把完整的结果存下来用于后续处理，可以直接 decode
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text

# ================= 2. 模型加载与启动 =================
def main():
    args = inference_args()
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
    checkpoint_path = f"{output_dir}/pretrain_epoch_{args.epochs - 1}.pth" 
    tokenizer_path = "./model" 
    
    # 2. 初始化模型架构 (需要和预训练时保持绝对一致)
    config = TinyuConfig(
        hidden_size=512, 
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    model, tokenizer = init_model(config, tokenizer_path, device=device)

    # 剥离 DDP 前缀
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    print("✅ 模型权重加载成功！\n")
    
    prompts = [
        "人工智能的未来发展趋势是",
        "def quick_sort(arr):"
    ]
    
    for p in prompts:
        generate_text_hf(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            max_new_tokens=150,
            temperature=0.7, 
            top_k=50,
            top_p=0.9,
            device=device
        )

if __name__ == "__main__":
    main()