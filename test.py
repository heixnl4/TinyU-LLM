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
print(torch.__version__) 
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))