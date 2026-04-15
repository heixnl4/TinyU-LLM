import json
import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


# 初始化预训练数据集
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        :param data_path: .jsonl 数据文件的路径
        :param tokenizer: 你的分词器实例
        :param max_length: 序列的最大截断长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回单条训练数据，包含 input_ids 和 labels
        """
        text = self.data[index]["text"]
        
        # ================= 2. 文本分词 (Tokenization) =================
        # padding="max_length" 保证所有样本长度绝对一致，这是打包成 Batch 的前提
        # truncation=True 保证超长文本会被截断
        # 如果预训练文本中有 <|im_end|> ，则 add_special_tokens=False
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"  # 直接返回 PyTorch 张量
        )
        # 形状从 [1, max_length] 变成 [max_length]
        input_ids = tokens["input_ids"].squeeze(0)
        
        # ================= 3. 构建标签 (Labels) =================
        # 语言模型的自回归特性：目标就是复述自己
        labels = input_ids.clone()
        
        # 屏蔽 Padding 部分的 Loss 计算，标签强制设为 -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return input_ids, labels
    
# tokenizer = AutoTokenizer.from_pretrained("../model")
# dataset = PretrainDataset("../dataset/pretrain_hq.jsonl", tokenizer)
# print("Raw text:", dataset.data[0]["text"]) 
# input_ids, labels = dataset[0]  
# print("input_ids:", input_ids)
# print("labels:", labels)
# print(len(dataset))

