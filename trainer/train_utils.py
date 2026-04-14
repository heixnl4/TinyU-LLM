import torch
from transformers import AutoTokenizer
from model.model_Tinyu import TinyuForcausalLM

def print_model_param_details(model, detail=False, prefix=""):
    print("=== Detailed Parameter Count ===")
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_count = param.numel()
        total_params += param_count
        if detail:
            print(f"{prefix}{name}: {param_count / 1e6:.3f}M")
    print(f"{'-'*50}")
    print(f"Total trainable params: {total_params / 1e6:.2f}M")
    print(f"{'-'*50}")

def init_model(config, tokenizer_path='../model'):
    model = TinyuForcausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    