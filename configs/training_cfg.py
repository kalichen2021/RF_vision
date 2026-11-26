

"""训练相关参数（数据管线 / 超参 / 学习率调度）。

用法::

    from configs.training_cfg import TrainingConfig

    TrainingConfig.BATCH_SIZE
"""

from __future__ import annotations


class TrainingConfig:
  """训练侧全部参数。"""

  # 训练用数据集："0" = 自动选最新；也可以写具体目录名
  SELECT_IMGSET_ID: str = "0"

  EPOCHS_HEAD: int = 15        # 阶段一：冻结 backbone，只训 head
  EPOCHS_FINETUNE: int = 25    # 阶段二：解冻 backbone，整体微调
  BATCH_SIZE: int = 48

  # --------数据管线（GPU 利用率低通常是这里没配好）--------
  # DataLoader 子进程数：0 = 主进程里读图（GPU 会等 CPU，利用率自然低）
  # 每个进程常驻约 0.6~1GB（torch + cv2），16G 内存的机器不要超过 4~6，
  # 否则会和训练主进程 / CUDA 抢内存，表现为主机内存暴涨、GPU 反而空转
  NUM_WORKERS: int = 10
  # 每个 worker 预取的 batch 数：越大越能掩盖读图抖动，但内存线性增长
  PREFETCH_FACTOR: int = 2
  # 锁页内存：配合 .to(non_blocking=True) 做异步 H2D 拷贝（仅 CUDA 有意义）
  PIN_MEMORY: bool = True
  # 混合精度训练（RTX 30 系 Tensor Core 收益明显；显存也更省）
  USE_AMP: bool = True
  # cudnn 自动选最快卷积算法；环境不稳时可能以 “CUDA error” 形式后面才炸，
  # 显存紧张 / 驱动不匹配时可改回 False
  CUDNN_BENCHMARK: bool = True
  # 主进程 torch 运算线程数（0 = 用 torch 默认 = CPU 逻辑核数）
  # 经验值：TORCH_NUM_THREADS + NUM_WORKERS ≈ 物理核数
  TORCH_NUM_THREADS: int = 4
  # 每个 DataLoader worker 内部线程数（读图 / Resize 不需要多线程）
  WORKER_TORCH_THREADS: int = 1

  TRAININGSET_RATIO: float = 0.8

  # --------训练超参--------
  DROPOUT: float = 0.3              # head 里的 Dropout2d 概率（0 表示不加）
  EARLY_STOPPING_PATIENCE: int = 4  # 连续多少轮监控指标没变好就停
  LR_HEAD: float = 1e-3             # 阶段一：冻结 backbone，只训 head
  LR_FINETUNE: float = 1e-4         # 阶段二：解冻 backbone（必须比阶段一小）

  # 每个 epoch 在验证集上跑一次：best 权重 / 早停 / 调度器全部以 val loss 为准
  # （训练 loss 受 Dropout2d 和 BN batch 统计影响，跟泛化能力不是一回事）
  VALIDATE_EVERY_EPOCH: bool = True
  # 每隔多少轮跑一次验证（1 = 每轮；>1 时只在验证轮更新 best / 早停 / 降 lr）
  VAL_INTERVAL: int = 1

  # 训练记录（每轮 train / val 指标）写到 dataset/log/train/<uuid>.json
  SAVE_TRAIN_LOG: bool = True

  # ReduceLROnPlateau：val loss 连续 LR_PLATEAU_PATIENCE 轮没变好就乘 LR_PLATEAU_FACTOR
  LR_PLATEAU_FACTOR: float = 0.5
  LR_PLATEAU_PATIENCE: int = 2
  LR_MIN: float = 1e-6              # 降 lr 的下限

  # 输入尺寸 (高, 宽)：训练、推理、摄像头必须一致
  IMG_SIZE: tuple[int, int] = (640, 640)