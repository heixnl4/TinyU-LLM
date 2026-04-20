import json
import os
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

def pre_processing_chat(conversations, add_system_ratio=0.2):
        if any(conv.get('tools') for conv in conversations): 
            return conversations

        SYSTEM_PROMPTS = [
            "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
            "你是 TinyU，一个小巧但强大的语言模型。",
            "You are a helpful AI assistant."
        ]
        
        if conversations[0].get('role') != 'system':
            if random.random() < add_system_ratio:
                return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
        return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
        # 防护机制：适当丢弃无意义的思考标签，增加模型输出格式的鲁棒性
        if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
        return prompt_content



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
    

class SFTDataset(Dataset):
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
        
        self.assistant_prefix = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.im_end_token = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        
        # 1. 获取基础对话列表
        conversations = sample['conversations']
        
        # 2. (可选) 预处理：按概率注入 System Prompt 等增强操作
        conversations = pre_processing_chat(conversations)
        
        # 3. 将对话结构转化为连续的纯文本长字符串
        prompt = self.create_chat_prompt(conversations)
        
        # 4. (可选) 后处理：动态清洗思考标签等
        prompt = post_processing_chat(prompt)
        
        # 5. Tokenize 整句文本并截断
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_length]
        
        # 6. 精准计算 Labels (必须在 Padding 之前计算)
        labels = self.generate_labels(input_ids)
        
        # 7. Padding (对齐长度)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids += [pad_token_id] * pad_len
            labels += [-100] * pad_len  # Pad 的部分也不计算 Loss
            
        # 8. 调试钩子 (极度建议在刚开始训练时打开，确认 Loss 的位置是对的)
        # if index == 0:
        #     self.debug_print(input_ids, labels)
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 处理 Tool 的反序列化
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
            
        # 使用 Hugging Face 底层 C++ 模板引擎进行渲染
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    
    def generate_labels(self, input_ids):
        """
        核心物理逻辑：滑动窗口匹配。
        只保留 Assistant 回答部分的 token，其余全设为 -100。
        """
        labels = [-100] * len(input_ids)
        prefix_len = len(self.assistant_prefix)
        input_len = len(input_ids)
        
        i = 0
        while i < input_len:
            # 找到 "<|im_start|>assistant\n" 的精确起手式
            if input_ids[i : i + prefix_len] == self.assistant_prefix:
                i += prefix_len
                start_idx = i
                
                # 往后找，直到遇到 "<|im_end|>"
                while i < input_len and input_ids[i] != self.im_end_token:
                    i += 1
                
                # 把 <|im_end|> 也包含在 Loss 中，教模型学会“闭嘴”
                end_idx = i + 1 if i < input_len else input_len 
                
                # 解除这段区间的掩码，暴露给交叉熵函数去算 Loss
                labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
            else:
                i += 1
                
        return labels
    
    # ================= 调试工具 =================
    def debug_print(self, input_ids, labels):
        print(f"\n{'='*20} 调试模式: 序列与标签映射确认 {'='*20}")
        for i, (x, y) in enumerate(zip(input_ids, labels)):
            # 只打印前 150 个 token，防止刷屏
            if i > 150: 
                break
            # y=-100 意味着该位置被 Mask 掉了，不计算梯度
            status = "❌ Masked (-100)" if y == -100 else f"✅ Compute Loss (Label: {y})"
            token_str = self.tokenizer.decode([x])
            print(f"[{i:3d}] Token: {token_str!r:15s} | {status}")
        print("="*65 + "\n")

    '''# 简易实现
    def __getitem__(self, index):
        """
        返回单条 SFT 训练数据，包含 input_ids 和 labels
        (核心区别：User 文本部分的 labels 会被掩盖为 -100)
        """
        conversations = self.data[index]["conversations"]
        
        input_ids = []
        labels = []
        
        # ================= 1. 多轮对话拼接与分词 =================
        for turn in conversations:
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user":
                text = "<|im_start|>user\n" + content + "<|im_end|>\n"
                # 必须设为 False，避免 tokenizer 自动在中间塞入 <s> 等句首符
                tokens = self.tokenizer(text, add_special_tokens=False).input_ids
                
                input_ids.extend(tokens)
                # User 的话不计算 Loss，强制设为 -100
                labels.extend([-100] * len(tokens))
                
            elif role == "assistant":
                text = "<|im_start|>assistant\n" + content + "<|im_end|>\n"
                tokens = self.tokenizer(text, add_special_tokens=False).input_ids
                
                input_ids.extend(tokens)
                # Assistant 的话是预测目标，拷贝原本的 token
                labels.extend(tokens)

        # ================= 2. 截断 (Truncation) =================
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        # ================= 3. 填充 (Padding) =================
        else:
            pad_len = self.max_length - len(input_ids)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # input_ids 用 pad_token_id 补齐
            input_ids.extend([pad_token_id] * pad_len)
            # labels 用 -100 补齐，保证 Pad 的空白部分也不参与 Loss 计算
            labels.extend([-100] * pad_len)

        # ================= 4. 返回张量 =================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)'''


class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_prompt_length):
        """
        PPO Prompt 数据集 (基于 HuggingFace datasets)
        :param data_path: jsonl 数据集路径
        :param tokenizer: 模型的 tokenizer
        :param max_prompt_length: prompt 的最大截断长度
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        # ================= 1. Tokenizer 预设 =================
        # PPO 阶段生成文本，必须保证 Padding 在左侧
        if self.tokenizer.padding_side != 'left':
            print("【警告】检测到 Tokenizer padding_side 不是 'left'！PPO 生成阶段强制修改为左侧填充。")
            self.tokenizer.padding_side = 'left'
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ================= 2. 加载数据集 =================
        # 使用 Hugging Face 的 load_dataset 读取 jsonl
        raw_dataset = load_dataset("json", data_files=data_path, split="train")

        # ================= 3. 数据处理函数 =================
        def process_fn(example):
            user_prompt = ""
            # 遍历 conversations，只提取 user 的提问
            # 丢弃原数据里 assistant 对应的 "空"，因为 PPO 需要模型自己生成
            for msg in example["conversations"]:
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break
            
            # 构造给模板使用的单轮对话格式
            chat_msg = [{"role": "user", "content": user_prompt}]
            
            # 使用模型的 Chat Template
            # add_generation_prompt=True 会自动在末尾加上助手生成的引导符
            prompt_text = self.tokenizer.apply_chat_template(
                chat_msg, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 对文本进行 Tokenize 并填充截断
            encoded = self.tokenizer(
                prompt_text,
                max_length=self.max_prompt_length,
                truncation=True,
                padding="max_length"
            )
            
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }

        # ================= 4. 批量映射与格式化 =================
        print("正在使用多线程处理 PPO Prompt 数据集...")
        # 使用 map 加速处理，处理完后删除原始的无关列（如原文本等），只保留需要的 tensor 字段
        self.dataset = raw_dataset.map(
            process_fn, 
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing prompts"
        )
        
        # 强制将 dataset 的输出转为 PyTorch Tensor 格式
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        print(f"成功加载并处理完成，共包含 {len(self.dataset)} 条有效数据。")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 由于上面已经使用了 set_format("torch")
        # 这里取出的直接就是字典格式的 Tensor， DataLoader 可以直接打包
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"]
        }