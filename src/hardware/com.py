import serial
import time
from .command import (move,
                      home,
                      set_zero,
                      set_enable,
                      soft_home,
                      describe,
                      send,
                      build_frame,
                      parse_all
                      )

from configs import PORT, BAUD, TIME_INTERVAL

def open_port(port, baud):
  global ser
  ser = serial.Serial(port, baud, timeout=0.2)
  time.sleep(0.2)
  ser.reset_input_buffer()
  print(f'已打开 {port} @ {baud}')
  return ser


class Gimbal:
  def __init__(self, port, baudrate=115200):
    self._enabled = False
    self._zeroed = False
    self._port = port
    self._baudrate = baudrate
    open_port(port, baudrate)

  def scan_all(self, scan_config):
    yaw_range = scan_config['yaw_range']
    pitch_range = scan_config['pitch_range']
    move('yaw', pitch_range[0])
    move('pitch', yaw_range[0])
    move('yaw', pitch_range[1])
    move('pitch', yaw_range[1])
  
  def backhome(self):
    home(0)
    self._zeroed = True
  
  def adjust_posi(self, yaw, pitch):
    r1 = move('yaw', yaw)
    time.sleep(TIME_INTERVAL)
    r2 = move('pitch', pitch)
    return [r1, r2]
