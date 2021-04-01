import board
import busio
import numpy as np
import adafruit_mlx90640


class IrCamera:
    RESOLUTION_X = 32
    RESOLUTION_Y = 24
    RESOLUTION = (RESOLUTION_Y, RESOLUTION_X)
    RESOLUTION_CV = (RESOLUTION_X, RESOLUTION_Y)

    def __init__(self, doubled_freq_hz: int):
        self._i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
        self._mlx = adafruit_mlx90640.MLX90640(self._i2c)
        self._fps = doubled_freq_hz / 2 / 1.02  # it is a little bit slower than the requested fps, maybe due to processing in the library

        freq_to_enum = {
            1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
            2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
            4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
            8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
            16: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ,
            32: adafruit_mlx90640.RefreshRate.REFRESH_32_HZ,
        }
        self._mlx.refresh_rate = freq_to_enum[doubled_freq_hz]

    def get_fps(self):
        return self._fps

    def get_frame(self):
        frame = np.zeros((IrCamera.RESOLUTION_Y * IrCamera.RESOLUTION_X))
        self._mlx.getFrame(frame)
        return frame
