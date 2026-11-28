from __future__ import annotations

"""硬件配置参数。"""


class HardwareConfig:
  # --------连接---------
  # 串口号
  SERIAL_PORT: str = "COM4"
  # 串口波特率
  SERIAL_BAUDRATE: int = 115200
  
  # ————————控制——————————
  TIME_INTERVAL: float = 0.05  # 控制循环间隔（单位：秒）
  YAW_RANGE: tuple[float, float] = (-60.0, 60.0)  # 云台偏航角度范围（单位：度）
  PITCH_RANGE: tuple[float, float] = (- 20.0, 20.0)  # 云台俯仰角度范围（单位：度）
