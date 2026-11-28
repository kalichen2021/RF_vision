import functools
import serial
import time
import struct
from functools import reduce
from operator import xor

HEAD1, HEAD2, TAIL = 0xAA, 0x55, 0x0D
CMD_YAW, CMD_PITCH = 0x01, 0x02
MAX_DATA = 0x40
CMD_HOME, CMD_SET_ZERO = 0x05, 0x06
CMD_ENABLE = 0x07
CMD_SOFT_HOME = 0x08


STATUS = {
    0x00: 'OK',
    0x01: 'ERR_LEN',
    0x02: 'ERR_RANGE',
    0x03: 'ERR_MOTOR',
    0x04: 'ERR_CMD',
    0x05: 'ERR_SELFTEST',
    0x06: 'ERR_ENABLE',
    0x07: 'ERR_SOFT_HOME',
}

_seq = 0


def next_seq():
  """序号 0x01~0xFF 循环"""
  global _seq
  _seq = _seq % 0xFF + 1
  return _seq


def checksum(data: bytes) -> int:
  """数据区逐字节异或（与 STM32 端实现一致）"""
  return reduce(xor, data, 0)


def build_frame(cmd: int, data: bytes, seq: int = None) -> bytes:
  if seq is None:
    seq = next_seq()
  assert 1 <= len(data) <= MAX_DATA, 'LEN 必须在 1~0x40'
  return bytes([HEAD1, HEAD2, cmd, seq, len(data)]) + data + bytes([checksum(data), TAIL])


def angle_to_data(deg: float) -> bytes:
  """角度(度) -> 2 字节大端，单位 0.1°；0° 与 360° 都写作 3600"""
  raw = int(round(deg * 10.0)) % 3600
  if raw == 0:
    raw = 3600
  return struct.pack('>H', raw)


def frame_yaw(deg, seq=None):
  return build_frame(CMD_YAW, angle_to_data(deg), seq)


def frame_pitch(deg, seq=None):
  return build_frame(CMD_PITCH, angle_to_data(deg), seq)


def hexs(b: bytes) -> str:
  return ' '.join(f'{x:02X}' for x in b)


print('yaw 90.0°  :', hexs(frame_yaw(90.0, seq=1)))
print('pitch 45.5°:', hexs(frame_pitch(45.5, seq=1)))


PORT = 'COM4'      # 改成设备管理器里 USB-TTL 的串口号
BAUD = 115200

ser = None


def parse_all(buf: bytearray):
  """从接收缓冲区取出所有完整帧，返回 [(cmd, seq, data)]，并就地删除已消费字节"""
  frames = []
  while True:
    start = -1
    for i in range(len(buf) - 1):
      if buf[i] == HEAD1 and buf[i + 1] == HEAD2:
        start = i
        break
    if start < 0:
      # 末尾可能是半个帧头，先留着
      if len(buf) >= 1 and buf[-1] == HEAD1:
        del buf[:-1]
      else:
        buf.clear()
      return frames

    del buf[:start]
    if len(buf) < 5:
      return frames

    ln = buf[4]
    if ln < 1 or ln > MAX_DATA:
      del buf[:2]
      continue

    total = 7 + ln          # 5 个头字段 + data + xor + tail
    if len(buf) < total:
      return frames

    data = bytes(buf[5:5 + ln])
    if buf[5 + ln] != checksum(data) or buf[6 + ln] != TAIL:
      del buf[:2]          # 坏帧，向后挪一个字节重新找帧头
      continue

    frames.append((buf[2], buf[3], data))
    del buf[:total]


def describe(frames):
  for cmd, seq, data in frames:
    status = data[0] if data else None
    name = STATUS.get(status, f'0x{status:02X}' if status is not None else '-')
    print(f'ACK cmd=0x{cmd:02X} seq={seq:3d} len={len(data):d} -> {name}')


# 自测：把应答帧自己解一遍
_buf = bytearray(bytes([HEAD1, HEAD2, 0x81, 0x01, 0x01, 0x00, 0x00, TAIL]))
describe(parse_all(_buf))





def send(frame: bytes, timeout=0.5):
  """发一帧并等待应答，返回 [(cmd, seq, data)]"""
  if ser is None:
    raise RuntimeError('请先 open_port()')
  ser.write(frame)
  ser.flush()
  print('TX:', hexs(frame))

  buf = bytearray()
  deadline = time.time() + timeout
  while time.time() < deadline:
    buf += ser.read(ser.in_waiting or 1)
    frames = parse_all(buf)
    if frames:
      return frames
  return []



def move(axis: str, deg: float, waiting=True, timeout=10.0):
  """axis: 'yaw' / 'pitch'，deg: 0~360 度"""
  frame = frame_yaw(deg) if axis == 'yaw' else frame_pitch(deg)
  ack = send(frame)
  if ack:
    describe(ack)
  else:
    print('!! 没收到应答：检查接线 / 波特率 / STM32 是否在跑协议模式')
  return ack


def move_both(deg: float):
  """命令字 0x00：两台电机一起动到同一角度

  固件会分别按 yaw / pitch 各自的增量下发位置命令，
  不依赖两台驱动器地址是否相同。
  """
  ack = send(build_frame(0x00, angle_to_data(deg)))
  if ack:
    describe(ack)
  else:
    print('!! 没收到应答：检查串口 / STM32 是否在跑协议模式')
  return ack


def can_selftest(timeout=1.0):
  """发 0x04 让固件切到 LOOPBACK 模式自发自收一帧（再自动切回正常模式）

  应答 0x84，STATUS:
    0x00 = 通过：单片机 CAN 外设 / 过滤器 / 中断 / 接收队列都正常
    0x05 = 不通过：问题在单片机侧（PB8/PB9 复用、AFIO 时钟、NVIC 中断）

  用途：区分「固件/单片机」与「外部收发器/接线」问题。
  自检期间约 50ms 电机会暂停响应，属正常现象。
  """
  frames = send(build_frame(0x04, bytes([0x00])), timeout=timeout)
  if not frames:
    print('!! 没收到自检应答：固件是不是还没更新到带 0x04 的版本？')
    return None

  cmd, seq, data = frames[0]
  if cmd != 0x84 or not data:
    print(f'!! 自检应答格式不对: cmd=0x{cmd:02X} len={len(data)}')
    return None

  status = data[0]
  if status == 0x00:
    print('CAN 回环自检: 通过 -> 单片机侧接收链路正常')
    print('   Bus-Off 就是外部问题：查收发器 RXD 是否接到 PB8、共地、CANH/CANL、120Ω')
  else:
    print(f'CAN 回环自检: 失败 (0x{status:02X}) -> 单片机侧接收链路异常')
    print('   查 PB8/PB9 remap、AFIO 时钟、NVIC 中断使能')
  return status


def query_status(timeout=0.5):
  """发 0x03 查询 CAN 诊断计数（固件 Gimbal.c 的 Gimbal_SendStatus）

  应答 0x83，20 字节大端（旧固件为 16 字节）：
    txFail(4) | busOff(4) | errCnt(4) | lastError(4) | ackMask(4)
  ackMask: bit0 = 地址 1(yaw) 回过应答，bit1 = 地址 2(pitch) 回过应答
    -> 若 ackMask 只有 0x01，说明 pitch 驱动器根本没回话
       （多半是它的 ID 还不是 2，或者没上电 / 没在 CAN 模式）
  """
  frames = send(build_frame(0x03, bytes([0x00])), timeout=timeout)
  if not frames:
    print('!! 没收到状态应答：固件可能是旧版本，或串口没通')
    return None

  cmd, seq, data = frames[0]
  if cmd != 0x83 or len(data) not in (16, 20):
    print(f'!! 应答格式不对: cmd=0x{cmd:02X} len={len(data)}')
    return None

  tx_fail, bus_off, err_cnt, last_err = struct.unpack('>IIII', data[:16])

  if len(data) >= 20:
    ack_mask = struct.unpack('>I', data[16:20])[0]
    ack_text = (f'0x{ack_mask:08X}  '
                f'yaw={"在线" if ack_mask & 0x1 else "无应答"}  '
                f'pitch={"在线" if ack_mask & 0x2 else "无应答"}')
  else:
    ack_text = '旧固件无此字段'

  info = {
      'txFail': tx_fail,
      'busOff': bus_off,
      'errCnt': err_cnt,
      'lastError': f'0x{last_err:08X}',
      'ackMask': ack_text,
  }
  print('CAN 诊断:', info)
  return info


# def port_test(func):
#   """装饰器：在函数前后自动 open_port() / ser.close()"""
#   @functools.wraps(func)
#   def wrapper(*args, **kwargs):
#     open_port()
#     try:
#       return func(*args, **kwargs)
#     finally:
#       ser.close()
#   return wrapper


def home(mode=0):
  """触发回零：0=单圈就近，1=单圈方向，2=多圈无限位碰撞，3=多圈有限位开关

  注意：回零是回到驱动器里**已存储的零点**。如果电机本来就在零点附近，
  不会看到任何动作——这是正常的，不是命令没发出去。
  """
  ack = send(build_frame(CMD_HOME, bytes([mode])))
  if ack:
    describe(ack)
  else:
    print('!! 没收到应答')
  return ack


def set_zero(store=True):
  """把当前位置设为零点

  store=True  -> 写入驱动器（掉电保存），装配/标定时做一次就够
  store=False -> 只在本次上电有效
  """
  ack = send(build_frame(CMD_SET_ZERO, bytes([1 if store else 0])))
  if ack:
    describe(ack)
  else:
    print('!! 没收到应答')
  return ack


def set_enable(on: bool):
  """使能/失能两台电机

  on=True  -> 上锁（闭环保持力矩，手拧不动）
  on=False -> 松开（可以自由拨动）
  """
  ack = send(build_frame(CMD_ENABLE, bytes([1 if on else 0])))
  if ack:
    describe(ack)
  else:
    print('!! 没收到应答')
  return ack


def soft_home():
  """软回零：用绝对位置模式回到 0，不依赖驱动器的回零流程（0x9A）

  用途：
    - home(0) 电机不动、但 soft_home() 能动  -> 驱动器拒绝了回零命令
    - 两者都不动                              -> 零点不在原处/被限位挡住
  """
  ack = send(build_frame(CMD_SOFT_HOME, bytes([0x00])))
  if ack:
    describe(ack)
  else:
    print('!! 没收到应答')
  return ack
