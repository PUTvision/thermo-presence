import datetime
import logging
import os
import threading
import time

# from matplotlib import pyplot as plt

import server
from data_collector import DataCollector
from devices.ir_camera import IrCamera
from devices.rgb_camera import RgbCamera
from ir_frame_collector import IrFrameCollector
import signal
import sys


logger = logging.getLogger(__name__)

data_collector = None  # type: DataCollector


def setup_logger():
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s ["%(module)s" - %(threadName)s]',
        level=logging.INFO,
        datefmt='%H:%M:%S')


def sig_int_handler(sig, frame):
    if sig_int_handler.already_called:
        print('\nCtrl+c received again. Exiting application immediately!')
        os._exit(-1)
    else:
        sig_int_handler.already_called = True
        print('\nCtrl+c received. Please wait for graceful application exit (or click ctrl+c again to kill)...')
        try:
            data_collector.finish_batch_recording()
        except:
            logger.exception("Failed to finish batch recording!")
        os._exit(-1)  # kill the app anyway, we don't have graceful tasks stopping


sig_int_handler.already_called = False


def main():
    global data_collector

    setup_logger()
    signal.signal(signal.SIGINT, sig_int_handler)

    ir_camera = IrCamera(doubled_freq_hz=4)
    ir_frame_collector = IrFrameCollector(ir_camera=ir_camera, video_zoom=8, save_video=True)
    rgb_camera = RgbCamera(fps=4, resolution=(1920//8, 1080//8))
    ir_frame_collector.start()

    data_collector = DataCollector(
        ir_frame_collector=ir_frame_collector,
        rgb_camera=rgb_camera)

    def run_server():
        server.run_server(ir_frame_collector=ir_frame_collector, rgb_camera=rgb_camera)

    threading.Thread(target=run_server).start()

    SINGLE_BATCH_RECORDING_DURATION = 4 * 60

    while True:
        logger.info("Batch iteration started...")
        try:
            data_collector.start_batch_recording()
            time.sleep(SINGLE_BATCH_RECORDING_DURATION)
            data_collector.finish_batch_recording()
        except:
            logger.exception("Error during batch recording!")
            time.sleep(1)


if __name__ == '__main__':
    main()
