

SELECT_IMGSET_ID = "0"

EPOCHS_HEAD = 15       # 阶段一：冻结 backbone，只训 head
EPOCHS_FINETUNE = 25   # 阶段二：解冻 backbone，整体微调
BATCH_SIZE = 48

# --------数据管线（GPU 利用率低通常是这里没配好）--------
# DataLoader 子进程数：0 = 主进程里读图（GPU 会等 CPU，利用率自然低）
# 每个进程常驻约 0.6~1GB（torch + cv2），16G 内存的机器不要超过 4~6，
# 否则会和训练主进程 / CUDA 抢内存，表现为主机内存暴涨、GPU 反而空转
NUM_WORKERS = 10
# 每个 worker 预取的 batch 数：越大越能掩盖读图抖动，但内存线性增长。
# batch=32 时单个 float32 batch 约 32*3*384*384*4 ≈ 57MB，prefetch 4 且
# workers=4 时仅排队就要 ~900MB（再加 pinned 拷贝会翻倍），容易 OOM
PREFETCH_FACTOR = 2
# 锁页内存：配合 .to(non_blocking=True) 做异步 H2D 拷贝（仅 CUDA 有意义）
PIN_MEMORY = True
# 混合精度训练（RTX 30 系 Tensor Core 收益明显；显存也更省）
USE_AMP = True
# cudnn 自动选最快卷积算法。注意：在 CUDA/driver 版本不匹配或显存紧张时，
# autotune 选到坏算法会以 “CUDA error: unknown error” 的形式在后面的 kernel 才报出来，
# 所以默认关掉；确认环境稳定后可改回 True 换取一点速度
CUDNN_BENCHMARK = True
# 主进程 torch 运算线程数（0 = 用 torch 默认 = CPU 逻辑核数）。
# 20 核机器上默认 14 线程会和 DataLoader worker 抢核，反而把 GPU 饿住；
# 经验值：TORCH_NUM_THREADS + NUM_WORKERS ≈ 物理核数
TORCH_NUM_THREADS = 4
# 每个 DataLoader worker 内部线程数（读图 / Resize 不需要多线程）
WORKER_TORCH_THREADS = 1

TRAININGSET_RATIO = 0.8

# 训练相关超参
DROPOUT = 0.3              # head 里的 Dropout2d 概率（0 表示不加；>0.5 会让 heatmap 收敛很慢）
EARLY_STOPPING_PATIENCE = 3  # 连续多少轮 loss 没变好就停
LR_HEAD = 1e-3             # 阶段一：冻结 backbone，只训 head
LR_FINETUNE = 1e-4         # 阶段二：解冻 backbone，整体微调（必须比阶段一小）

# 每个 epoch 在测试集上跑一次验证：best 权重 / 早停 / 调度器全部以 val loss 为准
# （训练 loss 受 Dropout2d 和 BN batch 统计影响，跟泛化能力不是一回事）
VALIDATE_EVERY_EPOCH = True

# ReduceLROnPlateau：val loss 连续 LR_PLATEAU_PATIENCE 轮没变好，就把 lr 乘 LR_PLATEAU_FACTOR
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
LR_MIN = 1e-6              # 降 lr 的下限