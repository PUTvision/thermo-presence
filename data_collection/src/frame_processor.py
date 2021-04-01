class FrameProcessor:
    def __init__(self):
        self._latest_frame = None

    def process_frame(self, frame):
        pass

    def get_latest_frame(self):
        return self._latest_frame
