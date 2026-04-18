import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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