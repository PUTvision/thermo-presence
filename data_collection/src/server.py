import logging
import time
from io import BytesIO

import cv2
import png
import numpy as np
from flask import Flask, Response, send_from_directory, redirect, request

from devices.rgb_camera import RgbCamera
from ir_frame_collector import IrFrameCollector


logger = logging.getLogger(__name__)


app = Flask(__name__, static_url_path='')


@app.route('/')
def get_root():
    return redirect('/index.html')


@app.route('/status')
def get_status():
    return "hello"


_ir_frame_collector = None  # type: IrFrameCollector
_rgb_camera = None  # type: RgbCamera


def _get_ir_frame(ir_frame_collector: IrFrameCollector, min_temp, max_temp):
    while True:
        raw_frame = ir_frame_collector.get_latest_frame().data
        # logger.info(f"Get frame with min/max={min_temp}/{max_temp}")
        video_frame_3d = ir_frame_collector.raw_frame_to_frame_for_video(
            raw_frame, as_bgr=False, min_temp=min_temp, max_temp=max_temp)
        video_frame_3d = cv2.rotate(video_frame_3d, cv2.ROTATE_90_CLOCKWISE)
        video_frame_3d = cv2.flip(video_frame_3d, 0)

        video_frame = video_frame_3d.reshape(video_frame_3d.shape[0], -1)
        frame_png = png.from_array(video_frame, 'RGB')
        output = BytesIO()
        frame_png.write(output)
        frame_bin = output.getvalue()

        time.sleep(0.1)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bin + b'\r\n')


def _get_processed_ir_frame(ir_frame_collector):
    while True:
        raw_frame = ir_frame_collector.frame_processor.get_latest_frame()

        # video_frame_3d = ir_frame_collector.raw_frame_to_frame_for_video(
        #     raw_frame, as_bgr=False, min_temp=min_temp, max_temp=max_temp)
        # video_frame = video_frame_3d.reshape(video_frame_3d.shape[0], -1)
        # frame_png = png.from_array(video_frame, 'RGB')
        # output = BytesIO()
        # frame_png.write(output)
        # frame_bin = output.getvalue()

        time.sleep(0.1)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bin + b'\r\n')


def _get_rgb_frame(rgb_camera: RgbCamera):
    while True:
        frame_bin = rgb_camera.get_frame_as_jpeg_bin()
        time.sleep(0.1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bin + b'\r\n')


@app.route('/ir_video')
def get_ir_video():
    min_temp = request.args.get('min', default=None, type=float)
    max_temp = request.args.get('max', default=None, type=float)

    global _ir_frame_collector
    return Response(
        _get_ir_frame(_ir_frame_collector, min_temp=min_temp, max_temp=max_temp),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/rgb_video')
def get_rgb_video():
    global _rgb_camera
    return Response(
        _get_rgb_frame(_rgb_camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/processed_ir_video')
def get_processed_ir_video():
    global _ir_frame_collector
    return Response(
        _get_processed_ir_frame(_ir_frame_collector),
        mimetype='multipart/x-mixed-replace; boundary=frame')


def run_server(ir_frame_collector, rgb_camera):
    logger.info("Starting server...")
    global _ir_frame_collector
    global _rgb_camera
    _ir_frame_collector = ir_frame_collector
    _rgb_camera = rgb_camera
    app.run(host='0.0.0.0', port=8888, debug=False)


@app.route('/index.html')
def get_index_html():
    return send_from_directory('', 'index.html')


if __name__ == '__main__':
    run_server(None, None)
