import os
from torch.distributed.run import main

if __name__ == "__main__":
    # 1. 在 Python 最顶层强行注入环境变量，100% 穿透到底层 C++
    os.environ["USE_LIBUV"] = "0"
    
    print("已强制关闭 libuv，正在启动分布式训练...")
    
    # 2. 直接调用 torchrun 的底层 main 函数，等价于在终端敲命令
    main(["--nproc_per_node=2", "train_pretrain_ddp_test.py"])