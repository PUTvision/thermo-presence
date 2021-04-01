import datetime
import logging
import os
import time

from devices.rgb_camera import RgbCamera
from ir_frame_collector import IrFrameCollector

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, rgb_camera: RgbCamera, ir_frame_collector: IrFrameCollector):
        self._rgb_camera = rgb_camera
        self._ir_frame_collector = ir_frame_collector

        root_dir_name = datetime.datetime.now().strftime("data__%d_%m_%Y__%H_%M_%S")
        self._root_dir_path = os.path.abspath(os.path.join('../data', root_dir_name))
        os.makedirs(self._root_dir_path, exist_ok=True)
        logger.info(f"Starting recording to root directory '{self._root_dir_path}'")
        self._data_batch_number = 0
        self._batch_subdir_path = ''

    def start_batch_recording(self):
        subdir_name = f"{self._data_batch_number:03}__" + datetime.datetime.now().strftime("%H_%M_%S")
        self._batch_subdir_path = os.path.join(self._root_dir_path, subdir_name)

        logger.info(f"Starting batch recording to subdirectory '{self._batch_subdir_path}'")
        os.makedirs(self._batch_subdir_path, exist_ok=True)
        rgb_video_path = os.path.join(self._batch_subdir_path, 'rgb.mjpeg')

        ir_video_path = os.path.join(self._batch_subdir_path, 'ir.avi')
        ir_csv_path = os.path.join(self._batch_subdir_path, 'ir.csv')

        self._ir_frame_collector.start_recording_frames(
            video_file_path=ir_video_path, csv_file_path=ir_csv_path)
        self._rgb_camera.start_recording_to_file(rgb_video_path)

    def finish_batch_recording(self):
        try:
            self._ir_frame_collector.stop_recording_frames()
            self._rgb_camera.stop_recording_to_file()
        except:
            logger.exception("Failed to finish recording batch!")

        self._data_batch_number += 1
