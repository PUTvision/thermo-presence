import numpy as np

import config


class IrDataCsvReader:
    def __init__(self, file_path):
        self._file_path = file_path
        self._frames, self._raw_frame_data = self.get_frames_from_file(file_path)

    @staticmethod
    def get_frames_from_file(file_path):
        frames = []
        raw_frame_data = []
        with open(file_path, 'r') as file:
            raw_lines = file.readlines()
        lines = [x.strip() for x in raw_lines]
        frame_lines = lines[1:]
        for frame_line in frame_lines:
            raw_frame_data.append(frame_line)
            line_parts = frame_line.split(',')
            frame_data_str = line_parts[2:]
            frame_data_1d = [float(x) for x in frame_data_str]
            frame_2d = np.reshape(frame_data_1d, config.IR_CAMERA_RESOLUTION)
            frames.append(frame_2d)
        return frames, raw_frame_data

    def get_number_of_frames(self):
        return len(self._frames)

    def get_frame(self, n):
        return self._frames[n]

    def get_raw_frame_data(self, n) -> str:
        return self._raw_frame_data[n]
