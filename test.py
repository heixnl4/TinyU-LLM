# D:\project\Tinyu\test_model.py
# from model.model_Tinyu import TinyuForcausalLM
# from model.configuration import TinyuConfig
# import torch

# config = TinyuConfig(
#     hidden_size=128,
#     num_hidden_layers=4,
#     num_attention_heads=2,
#     num_key_value_heads=1,
#     use_moe=True,
#     vocab_size=1000
# )

# model = TinyuForcausalLM(config)
# input_ids = torch.randint(0, 1000, (1, 128))
# output = model(input_ids)

# print("Logits shape:", output.logits.shape)
# print("Aux loss:", output.aux_loss)
# print("Hidden states shape:", output.hidden_states.shape)


import torch
# print(torch.__version__) 
# print("torch.cuda.is_available():", torch.cuda.is_available())
# print("torch.version.cuda:", torch.version.cuda)
# print("Number of GPUs:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("GPU name:", torch.cuda.get_device_name(0))


# 查看模型实打实到底用了多少显存（通常远低于 2GB）
print(f"真实使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 查看 PyTorch 向显卡总共申请了多少显存（接近你看到的 2GB）
print(f"缓存池预留: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# tokenizer = AutoTokenizer.from_pretrained("../model")
# dataset = PretrainDataset("../dataset/pretrain_hq.jsonl", tokenizer)
# print("Raw text:", dataset.data[0]["text"]) 
# input_ids, labels = dataset[0]  
# print("input_ids:", input_ids)
# print("labels:", labels)
# print(len(dataset))
