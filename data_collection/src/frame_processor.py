class FrameProcessor:
    def __init__(self):
        self._latest_frame = None

    def process_frame(self, frame):
        # TODO - at some live processing here...
        pass

    def get_latest_frame(self):
        return self._latest_frame
