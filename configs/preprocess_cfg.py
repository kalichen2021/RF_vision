"""离线数据合成（预处理）参数。

用法::

    from configs.preprocess_cfg import PreprocessConfig

    PreprocessConfig.AREA_RANGE
"""

from __future__ import annotations


class PreprocessConfig:
  """合成样本生成的全部参数。"""

  # 本次批量生成要产出的样本总数（`-n` 可临时覆盖）
  SAMPLE_GEN_NUM: int = 20000
  # 落盘前先缩放到这个尺寸 (高, 宽)：训练输入本来就是 384，
  # 存 384 比存 720 的 PNG **解码快 3~4 倍**（读图是训练数据管线的大头）；
  # None = 保持原图尺寸（以后想换 512 训练时才需要）
  SAVE_SIZE: tuple[int, int] | None = (640, 640)
  # 样本图格式："png"（无损）或 "jpg"（解码更快、体积更小，但轻微压缩）
  SAVE_EXT: str = "png"
  # 数字类别数（贴图池覆盖 0 ~ N-1，需要 >= 10 才能覆盖 0~9）
  SAMPLE_COUNT_RATE: int = 10
  # 每张样本包含的数字个数范围（含两端）；凑不够下限就换一张图重试
  SAMPLE_DIGIT_RANGE: tuple[int, int] = (1, 7)

  # --------环境图采样（避免单张图被反复使用）---------
  # 单张环境图被成功使用的次数达到该值后开始降权
  ENV_USE_SOFT_CAP: int = 4
  # 超过阈值后每多用一次，权重乘以该衰减系数（越小越不容易被再次选中）
  ENV_USE_DECAY: float = 0.2

  # 环境图先**裁成正方形**（起点随机）再贴数字：样本本身就是方形，
  # 训练时缩放不会把画面拉扁，同时随机起点也等于一次位置增强
  ENV_CROP_SQUARE: bool = True

  # --------QUADS---------（目标：模拟「板上写着数字」的平面放置）
  # 畸变容忽度（映射为允许长宽比 1.5 + 3.5*d），板子接近矩形，取偏小值。
  # 注意：这只控制「允许的格子长宽比」，**不是透视强度**。
  DISTORTION: tuple[float, float] = (0.0, 0.01)

  # --------板面姿态（透视强度 + 三轴旋转）---------
  # 以前板面四边形直接继承环境图里检测到的平坦区域形状，所以照片拍得越近、
  # 透视就越强，而且没有任何参数可以调。下面这组参数会把板面形状按
  # 「等效相机距离 + 三轴倾角」重新算一遍（实现在 src/preprocess/perspective.py）：
  #   只改板面四边形的形状，中心/面积基本不变，标签由它派生，格式不变。
  POSE_RESAMPLE: bool = True
  # 相机距离 / 板面特征尺寸 sqrt(面积)：越大透视越弱（灭点越远、越像长焦），
  # 越小透视越强（近距广角）。板面接近正方形时，≈3 明显透视，≈10 近似正交。
  POSE_DISTANCE: tuple[float, float] = (7.5, 9.0)
  # 三轴倾角（度）：X = 俯仰（上下翻）、Y = 偏航（左右转）、Z = 面内滚转
  POSE_ROT_X: tuple[float, float] = (-25, 25)
  POSE_ROT_Y: tuple[float, float] = (-70, 70)
  POSE_ROT_Z: tuple[float, float] = (-20, 20)
  # 重采样后的面积倍率（1.0 = 与原板面面积一致）
  POSE_AREA_JITTER: tuple[float, float] = (0.9, 1.1)
  # 重采样后如何决定面积（决定“转过去”看不看得出来）：
  #   "keep"    = 把面积拉回原板面（默认）。板面大小稳定，但大角度偏航/俯仰会被
  #               面积补偿抵消，视觉上只剩轻微剪切。
  #   "natural" = 不做面积补偿，保留透视压缩：转到 70° 时板面真的变窄变偏。
  #               ⚠️ 此时板面/数字面积会随角度变化，AREA_RANGE 不再是硬约束。
  POSE_AREA_MODE: str = "natural"
  # 0 = 完全沿用原形状，1 = 完全用新形状；若觉得板子“像贴上去的”，调到 0.6~0.8
  POSE_BLEND: float = 1.0

  # --------生成速度---------
  # 并行线程数（1 = 单线程）。融合 / 切分是 CPU 密集，线程数 ≈ 物理核数即可；
  # 内存很紧时降到 2~4（每个在途任务会持有一张 720×720 的缓冲区）
  GEN_WORKERS: int = 8
  # 字形池倍数：一次性预渲染 SAMPLE_COUNT_RATE * FONT_VARIANTS * 该值 个字，
  # 不再每个样本都用 PIL 重新渲染（原先这是最大的单笔开销）
  GLYPH_POOL_FACTOR: int = 4
  # 网格弯曲度：板是刚体，必须为 0，否则切出来的格子不是平面四边形的透视投影
  NORMALIZE: tuple[float, float] = (0.0, 0.0)
  # 板位（板面）四边形面积（像素²）：<1 时按占平坦区域面积的比例解释；
  # 数字面积 = 板位面积 × (1 - 2*留边_横) × (1 - 2*留边_纵)，换留边不会改变板位大小
  AREA_RANGE: tuple[int, int] = (8000, 16000)
  # 笔画粗细 / 画布尺寸的候选，用来增加字形多样性
  THICKNESS_CHOICES: tuple[int, ...] = (0, 0, 1, 1, 2, 3, 3, 4, 4)
  CANVAS_CHOICES: tuple[int, ...] = (48, 64, 64, 80)
  # 每个数字渲染几个字形变体（不同字体 / 字重 / 画布）
  FONT_VARIANTS: int = 5

  # 从候选板位里选多少块真放数字（仅当 SAMPLE_DIGIT_RANGE 为 None 时作为旧逻辑使用）
  QUADS_SELECT_RATE: float = 0.5

  # --------FUSE / DEFOCUS（数字融合 + 景深失焦）---------
  # 数字与环境的对比度区间（|数字 - 背景| / 255），越大数字越醒目
  CONTRAST_RANGE: tuple[float, float] = (0.35, 0.90)
  # 数字与环境的融合透明度区间，越大数字越「实」
  FUSE_ALPHA: tuple[float, float] = (0.55, 0.95)
  # 失焦强度倍率：0 = 全部清晰，越大非基准平面的数字越糊
  DEFOCUS_STRENGTH: float = 0.3
  # 单块数字贴图的最大高斯 sigma（像素），同时是失焦封顶值
  DEFOCUS_MAX_SIGMA: float = 4.0
  # 同一块数字内部「近端 → 远端」的失焦差异比例（0 = 均匀失焦）
  DEFOCUS_PERSPECTIVE: float = 0.5
  # 失焦尺寸封顶：sigma <= DEFOCUS_SIZE_RATIO * 数字短边，保证远处小数字仍可辨认
  DEFOCUS_SIZE_RATIO: float = 0.06

  # --------BOARD（板面渲染）---------
  # 板面**绝对**目标灰度区间（不是相对背景的偏移）：以白纸 / 白板为主，
  # 数字会按对比度自动压暗 → 稳定得到「白纸黑字」（暗环境也一样）
  BOARD_GRAY: tuple[int, int] | None = (190, 240)
  # 反极性（深色板 + 亮字）样本比例：0 = 全部白纸黑字；以白纸黑字为主时取小值
  POLARITY_FLIP_RATE: float = 0
  # 反极性样本的板面灰度区间（深色板）
  BOARD_GRAY_FLIP: tuple[int, int] = (25, 70)
  # 融合后最小可见对比度 |数字-背景|*alpha/255：低于该值就拉满透明度并提高对比度，
  # 避免出现「几乎看不见但仍有标签」的噪声样本（会抬高 loss 地板）
  MIN_VISIBLE_CONTRAST: float = 0.12
  # 旧参数：板面相对背景的亮度偏移，仅当 BOARD_GRAY 为 None 时生效
  BOARD_TONE: tuple[int, int] = (12, 30)
  # 数字在板面内的留边（内缩）比例，**宽、高分别给区间**：
  #   ((横_最小, 横_最大), (纵_最小, 纵_最大))
  # 每张图横/纵各抽一档，同一张图内所有数字共用；0 = 数字在该方向铺满板面。
  # 也兼容旧写法：写单个数字 = 横纵固定值，写 (lo, hi) = 横纵同一个区间。
  BOARD_MARGIN: tuple[tuple[float, float], tuple[float, float]] = (
      (0.3, 0.36), (0.33, 0.39))
  # 是否给板面画软投影（能提供更强的平面 / 遮挡线索）
  BOARD_SHADOW: bool = False
  # 判定「同一平面」的深度容差（0~1）
  PLANE_TOL: float = 0.12
  # 基准平面的深度；None 表示按深度最集中的一簇自动挑选
  BASE_DEPTH: float | None = None
