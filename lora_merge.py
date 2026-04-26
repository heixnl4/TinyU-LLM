from trainer.lora_utils import merge_lora_weights 

if __name__ == "__main__":
    # 这里的路径请替换为你真实的路径
    BASE_MODEL_PATH = "./tinyu_out/run-default_h512_l4_ah8_moe0/pretrain_weight.pth"
    LORA_MODEL_PATH = "./sft_out/lora_epoch_2.pth"
    OUTPUT_PATH = "./merged_out/tinyu_sft_merged.pth"
    
    # ⚠️ 必须和你在 SFT 训练时使用的 r 和 alpha 绝对一致！
    LORA_RANK = 8
    LORA_ALPHA = 32.0
    
    merge_lora_weights(
        base_model_path=BASE_MODEL_PATH,
        lora_model_path=LORA_MODEL_PATH,
        output_path=OUTPUT_PATH,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA
    )