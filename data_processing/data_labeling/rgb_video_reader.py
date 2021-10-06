import logging

import cv2


class RgbVideoReader:
    def __init__(self, file_path):
        self._file_path = file_path
        self._video_cap = None
        self._frames = self.get_frames_from_file(file_path)

    @staticmethod
    def get_frames_from_file(file_path):
        video_cap = cv2.VideoCapture(file_path)
        if not video_cap.isOpened():
            raise Exception("Video open failed!")

        frames = []
        while True:
            flag, frame = video_cap.read()
            if flag:
                frames.append(frame)
                if len(frames) % 50 == 0:
                    logging.debug(f"Reading RGB frame no {len(frames)}")
            else:
                break

        video_cap.release()
        return frames

    def get_number_of_frames(self):
        return len(self._frames)

    def get_frame(self, frame_index):
        return self._frames[frame_index]
