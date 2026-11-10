import serial

ser = serial.Serial("COM3", 115200, timeout=1)
ser.write(bytes([0xAA, 0x55, 0x01, 0x00, 0x00]))
data = ser.read(10)
