import cv2
import numpy as np

import config
from data_labeling import labeling_config


def x_on_interpolated_image_to_raw_x(x):
    x_raw_flipped = x / labeling_config.IR_FRAME_RESIZE_MULTIPLIER
    x_raw = config.IR_CAMERA_RESOLUTION_X - 1 - x_raw_flipped
    return x_raw


def y_on_interpolated_image_to_raw_y(y):
    return y / labeling_config.IR_FRAME_RESIZE_MULTIPLIER


def xy_on_interpolated_image_to_raw_xy(xy: tuple) -> tuple:
    return (x_on_interpolated_image_to_raw_x(xy[0]),
            y_on_interpolated_image_to_raw_y(xy[1]))


def x_on_raw_image_to_x_on_interpolated_image(x):
    x_flipped = config.IR_CAMERA_RESOLUTION_X - 1 - x
    return round(x_flipped * labeling_config.IR_FRAME_RESIZE_MULTIPLIER)


def y_on_raw_image_to_y_on_interpolated_image(y):
    return round(y * labeling_config.IR_FRAME_RESIZE_MULTIPLIER)


def xy_on_raw_image_to_xy_on_interpolated_image(xy: tuple) -> tuple:
    return (x_on_raw_image_to_x_on_interpolated_image(xy[0]),
            y_on_raw_image_to_y_on_interpolated_image(xy[1]))


def get_extrapolated_ir_frame_heatmap_flipped(
        frame_2d, multiplier, interpolation, min_temp, max_temp, colormap):
    new_size = (frame_2d.shape[1] * multiplier, frame_2d.shape[0] * multiplier)
    frame_resized_not_clipped = cv2.resize(
        src=frame_2d, dsize=new_size, interpolation=interpolation)

    if min_temp is None:
        min_temp = min(frame_resized_not_clipped.reshape(-1))
    if max_temp is None:
        max_temp = max(frame_resized_not_clipped.reshape(-1))

    frame_resized = np.clip(frame_resized_not_clipped, min_temp, max_temp)
    frame_resized_normalized = (frame_resized - min_temp) * (255 / (max_temp - min_temp))
    frame_resized_normalized_u8 = frame_resized_normalized.astype(np.uint8)
    heatmap_u8 = (colormap(frame_resized_normalized_u8) * 2 ** 8).astype(np.uint8)[:, :, :3]
    heatmap_u8_bgr = cv2.cvtColor(heatmap_u8, cv2.COLOR_RGB2BGR)
    heatmap_u8_bgr_flipped = cv2.flip(heatmap_u8_bgr, 1)  # horizontal flip
    return heatmap_u8_bgr_flipped
