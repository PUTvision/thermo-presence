import time

from data_collection.src.devices.ir_camera import IrCamera


def timed(f):
    """ Measure function execution time
    Just add decorator to the function
    """
    def wrap(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f'# {f.__name__}: {duration_ms:.3f} ms')
        return result
    return wrap


ir_camera = RgbCamera()


@timed
def get_single_frame():
    ir_camera.get_frame()


if __name__ == '__main__':
    while 1:
        get_single_frame()

"""
At refresh rate of 8Hz:

# get_single_frame: 254.683 ms
# get_single_frame: 254.821 ms
# get_single_frame: 254.745 ms
# get_single_frame: 254.922 ms
# get_single_frame: 254.767 ms

because 2 frames are required for one frame in chess mode

At 16Hz:
# get_single_frame: 127.555 ms
# get_single_frame: 127.589 ms
# get_single_frame: 127.608 ms
# get_single_frame: 127.638 ms
# get_single_frame: 127.382 ms

"""
