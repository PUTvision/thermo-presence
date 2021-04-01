import copy
import logging
import time
from typing import List, Union
from dataclasses import dataclass
import queue
import io

import cv2
import numpy as np
import threading
from matplotlib import pyplot as plt

# from devices.ir_camera import IrCamera
from frame_processor import FrameProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class IrFrame:
    data: np.ndarray
    timestamp: float


class IrFrameCollector:
    def __init__(self, ir_camera, video_zoom: int, save_video: bool):
        self._frames_queue = queue.Queue()
        self._access_lock = threading.RLock()
        self._record_frames = False
        self._ir_camera = ir_camera  # type: IrCamera
        self._reading_thread = threading.Thread(target=self._run_reading, name=self.__class__.__name__ + '_read')
        self._writing_thread = threading.Thread(target=self._run_writing, name=self.__class__.__name__ + '_write')
        self._colormap = plt.get_cmap('inferno')  # default for imshow is 'viridis'
        self._latest_frame = None
        self._zoomed_frame_size = (self._ir_camera.RESOLUTION_Y * video_zoom, self._ir_camera.RESOLUTION_X * video_zoom)
        self._zoomed_frame_size_cv = (self._zoomed_frame_size[1], self._zoomed_frame_size[0])
        self._save_video = save_video
        self._csv_file = None  # type: io.TextIOWrapper
        self._video_writer = None
        self._number_of_frames_written_to_csv_file = 0
        self.frame_processor = FrameProcessor()

    def start(self):
        self._reading_thread.start()
        self._writing_thread.start()

    def get_latest_frame(self) -> Union[IrFrame, None]:
        with self._access_lock:
            return copy.deepcopy(self._latest_frame)

    def start_recording_frames(self, video_file_path, csv_file_path):
        logger.info("About to start_recording_frames")

        self._csv_file = open(csv_file_path, 'wt')
        self._csv_file.write("no, timestamp, data_array\n")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # try also: MJPG and .avi
        fps = self._ir_camera.get_fps()
        self._video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, self._zoomed_frame_size_cv)
        self._record_frames = True

    def stop_recording_frames(self):
        self._record_frames = False

        self._frames_queue.join()  # wait until queue is empty
        logger.info('frames queue empty - stopped!')
        self._csv_file.close()
        self._video_writer.release()
        self._number_of_frames_written_to_csv_file = 0

    def _run_writing(self):
        logger.info("_run_writing started")
        while True:
            frame = self._frames_queue.get()
            # logger.debug("_run_writing - next frame!")
            try:
                data_str = ','.join([f"{t:.2f}" for t in frame.data])
                self._csv_file.write(f'{self._number_of_frames_written_to_csv_file},'
                                     f'{frame.timestamp},'
                                     f'{data_str}\n')
                self._number_of_frames_written_to_csv_file += 1

                final_video_frame = self.raw_frame_to_frame_for_video(
                    frame.data, min_temp=18, max_temp=35)
                self._video_writer.write(final_video_frame)

                self.frame_processor.process_frame(frame.data)
            except:
                logger.exception('Frame writing iteration error!')
            self._frames_queue.task_done()

    def _run_reading(self):
        logger.info("_run_reading started")
        previous_frame_time = time.time()
        while True:
            try:
                new_frame = IrFrame(
                    data=self._ir_camera.get_frame(),
                    timestamp=time.time(),
                )

                new_time = time.time()
                logger.debug(f"New IR frame collected. Diff =   "
                             f"{int((new_time-previous_frame_time)*1000)}")
                previous_frame_time = new_time

                with self._access_lock:
                    self._latest_frame = new_frame

                if not self._record_frames:
                    continue
                self._frames_queue.put(new_frame)
            except:
                logger.exception("Failed to acquire new IR frame!")
                time.sleep(0.1)  # to not spam with log if it fails all the time

    def raw_frame_to_frame_for_video(self, raw_frame, min_temp=None, max_temp=None, as_bgr=True):
        frame_2d = np.reshape(raw_frame, self._ir_camera.RESOLUTION)
        frame_resized_not_clipped = cv2.resize(src=frame_2d, dsize=self._zoomed_frame_size_cv, interpolation=cv2.INTER_CUBIC)

        if min_temp is None:
            min_temp = min(raw_frame)
        if max_temp is None:
            max_temp = max(raw_frame)
        # logger.debug(f"raw_frame_to_frame_for_video: min={min_temp}, max={max_temp}")

        frame_resized = np.clip(frame_resized_not_clipped, min_temp, max_temp)

        frame_resized_normalized = (frame_resized - min_temp) * (255 / (max_temp - min_temp))
        frame_resized_normalized_u8 = frame_resized_normalized.astype(np.uint8)
        heatmap_u8 = (self._colormap(frame_resized_normalized_u8) * 2**8).astype(np.uint8)[:, :, :3]

        if as_bgr:
            heatmap_u8_bgr = cv2.cvtColor(heatmap_u8, cv2.COLOR_RGB2BGR)
            return heatmap_u8_bgr
            # heatmap_u8_rgb = heatmap_u8_bgr[..., ::-1]
        else:
            return heatmap_u8

        # plt.imshow(frame_resized_normalized, norm=matplotlib.colors.Normalize(vmin=0, vmax=255))
        # plt.show()
        # cv2.imshow('frame_resized_normalized', final_frame)
        # cv2.waitKey(0)
