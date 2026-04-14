import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.configuration import TinyuConfig
from trainer.train_utils import print_model_param_details, init_model
config = TinyuConfig(
    hidden_size=256, 
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    use_moe=True
)

model, tokenizer = init_model(config, device='cuda')

print_model_param_details(model)