import time
import io
import numpy as np

import picamera
from picamera.array import PiRGBArray


class RgbCamera:
    def __init__(self, fps, resolution):
        self._camera = picamera.PiCamera()
        self._camera.resolution = resolution
        self._camera.framerate = fps
        # self._camera.start_preview()
        time.sleep(0.5)  # warm up

    def start_recording_to_file(self, file_path):
        self._camera.start_recording(file_path)

    def stop_recording_to_file(self):
        self._camera.stop_recording()

    def get_frame_as_array(self) -> np.ndarray:
        raw_capture = PiRGBArray(self._camera)
        self._camera.capture(raw_capture, format="rgb", use_video_port=True)
        image = raw_capture.array
        return image

    def get_frame_as_jpeg_bin(self) -> bytes:
        data_stream = io.BytesIO()
        self._camera.capture(data_stream, 'jpeg', use_video_port=True)
        return data_stream.getvalue()
