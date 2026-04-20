import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ================= 1. LoRA 线性层核心实现 =================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, lora_alpha=32, lora_dropout=0.1):
        """
        用这个类把原来的 nn.Linear 包裹起来。
        """
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # 缩放因子：保证即便改变了 r，初始化时的梯度大小也大致稳定
        self.scaling = self.lora_alpha / self.r

        # 1. 挂载并冻结原始的 Linear 层
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # 2. Dropout 防过拟合
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # 3. 定义低秩矩阵 A 和 B
        # 注意要继承原线性层的设备(device)和数据类型(dtype，比如 bfloat16)
        factory_kwargs = {'device': original_linear.weight.device, 'dtype': original_linear.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))

        # 4. 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # A 采用 Kaiming 均匀分布初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 必须全零初始化，保证初始状态下 LoRA 分支对模型毫无影响
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # 原始主路的输出 (完全被冻结，当做常量)
        base_out = self.original_linear(x)
        
        # LoRA 旁路的输出
        # PyTorch 的 F.linear(x, weight) 实际上算的是 x @ weight^T
        # 所以这里的连乘逻辑是：x 先过 dropout，乘 A^T，再乘 B^T，最后乘以缩放因子
        lora_out = F.linear(self.dropout(x), self.lora_A)  # [batch, seq, r]
        lora_out = F.linear(lora_out, self.lora_B)         # [batch, seq, out_features]
        lora_out = lora_out * self.scaling
        
        # 主路 + 旁路 = 最终输出
        return base_out + lora_out

# ================= 2. LoRA 注入机制 =================
def inject_custom_lora(model: nn.Module, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], r=8, lora_alpha=32, lora_dropout=0.1):
    """
    遍历模型，把目标层的 nn.Linear 替换为我们的 LoRALinear。
    """
    # 先把整个模型的所有参数全部冻结！
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    
    # 遍历模型所有的子模块
    for name, module in model.named_modules():
        # 比如 name 可能是 "layers.0.attention.q_proj"
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            # 找到目标的父模块 (比如 "layers.0.attention")
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            
            # 实例化我们手写的 LoRA 层
            lora_layer = LoRALinear(
                original_linear=module, 
                r=r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout
            )
            
            # 替换掉父模块里的旧 Linear 层
            setattr(parent_module, child_name, lora_layer)
            replaced_count += 1
            
    print(f"成功将 {replaced_count} 个 {target_modules} 层替换为自定义 LoRA 架构！")
    return model

# ================= 3. 参数统计工具 =================
def print_trainable_parameters(model: nn.Module):
    """
    打印当前可训练参数的占比，让你直观感受 LoRA 的恐怖缩模率。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # 如果是 float16/bfloat16，参数量不变，只是体积减半，所以用 numel 直接算个数
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    print(f"📊 可训练参数: {trainable_params:,d} || 全部参数: {all_param:,d} || 可训练比例: {100 * trainable_params / all_param:.4f}%")



def merge_lora_weights(base_model_path, lora_model_path, output_path, r=8, lora_alpha=32.0):
    """
    将 LoRA 权重离线合并回预训练基座权重中。
    
    :param base_model_path: 预训练基座权重的路径 (.pth)
    :param lora_model_path: SFT 训练得到的纯 LoRA 权重路径 (.pth)
    :param output_path: 合并后新权重的保存路径
    :param r: 训练 LoRA 时使用的 rank
    :param lora_alpha: 训练 LoRA 时使用的 alpha
    """
    print("开始合并 LoRA 权重...")
    
    # 1. 加载基座权重 (放到 CPU 上进行，防止撑爆显存)
    print(f"正在加载基座权重: {base_model_path}")
    base_state_dict = torch.load(base_model_path, map_location="cpu")
    
    # 清洗 DDP 带来的 "module." 前缀（如果有的话）
    base_state_dict = {
        k.replace("module.", "") if k.startswith("module.") else k: v 
        for k, v in base_state_dict.items()
    }
    
    # 2. 加载 LoRA 权重
    print(f"正在加载 LoRA 权重: {lora_model_path}")
    lora_state_dict = torch.load(lora_model_path, map_location="cpu")
    lora_state_dict = {
        k.replace("module.", "") if k.startswith("module.") else k: v 
        for k, v in lora_state_dict.items()
    }

    # 计算缩放因子
    scaling = lora_alpha / r
    print(f"LoRA 缩放因子 (alpha/r): {scaling}")

    # 3. 提取所有的 LoRA 目标层前缀
    # lora_state_dict 里的键名类似: "layers.0.attention.q_proj.lora_A"
    # 我们要提取出 "layers.0.attention.q_proj" 作为归类标识
    lora_keys = set()
    for key in lora_state_dict.keys():
        if "lora_A" in key or "lora_B" in key:
            # 剥离最后的 .lora_A 或 .lora_B
            base_key = key.replace(".lora_A", "").replace(".lora_B", "")
            lora_keys.add(base_key)

    # 4. 执行矩阵合并
    merged_state_dict = OrderedDict(base_state_dict)
    merged_count = 0
    
    for base_key in lora_keys:
        lora_A_key = f"{base_key}.lora_A"
        lora_B_key = f"{base_key}.lora_B"
        
        # ⚠️ 避坑：基座权重的键名通常是 "xxx.weight"
        # 但在我们的手写注入逻辑中，原来叫 weight 的参数被包装到了 original_linear.weight 里
        # 为了通用性，我们直接定位合并后要写回的键名，通常基座里是 base_key + ".weight"
        target_base_weight_key = f"{base_key}.weight"
        
        # 如果找不到标准的 weight 键名，可能因为基座本身有前缀，做个兼容检查
        if target_base_weight_key not in merged_state_dict:
            # 尝试找带 original_linear 的
            alt_key = f"{base_key}.original_linear.weight"
            if alt_key in merged_state_dict:
                target_base_weight_key = alt_key
            else:
                print(f"找不到 {base_key} 对应的基座权重，跳过合并。")
                continue

        # 提取张量
        # 将所有张量升频到 FP32 进行高精度乘加计算，计算出完美的数值后，再降回 BF16/FP16 覆盖回去
        lora_A_tensor = lora_state_dict[lora_A_key].to(torch.float32)
        lora_B_tensor = lora_state_dict[lora_B_key].to(torch.float32)
        base_weight_tensor = merged_state_dict[target_base_weight_key].to(torch.float32)

        # 核心数学操作：W_new = W_base + (B @ A) * scaling
        # PyTorch 的 Linear 权重形状是 [out_features, in_features]
        # A 是 [r, in_features], B 是 [out_features, r]
        # 所以 B @ A 的结果刚好是 [out_features, in_features]
        delta_weight = (lora_B_tensor @ lora_A_tensor) * scaling
        
        # 加和并恢复回原来的精度类型 (比如 bfloat16 或 float16)
        merged_weight = (base_weight_tensor + delta_weight).to(merged_state_dict[target_base_weight_key].dtype)
        
        # 更新字典
        merged_state_dict[target_base_weight_key] = merged_weight
        merged_count += 1

    # 5. 保存合并后的全新基座模型
    print(f"成功合并了 {merged_count} 个权重矩阵！")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"正在保存合并后的模型至: {output_path}")
    torch.save(merged_state_dict, output_path)
    print("合并完成！")
