

SELECT_IMGSET_ID = "0"

EPOCHS_HEAD = 10       # 阶段一：冻结 backbone，只训 head
EPOCHS_FINETUNE = 25   # 阶段二：解冻 backbone，整体微调
BATCH_SIZE = 4

TRAININGSET_RATIO = 0.8

# 训练相关超参
DROPOUT = 0.3              # head 里的 Dropout2d 概率（0 表示不加；>0.5 会让 heatmap 收敛很慢）
EARLY_STOPPING_PATIENCE = 3  # 连续多少轮 loss 没变好就停
LR_HEAD = 1e-3             # 阶段一：冻结 backbone，只训 head
LR_FINETUNE = 1e-4         # 阶段二：解冻 backbone，整体微调（必须比阶段一小）