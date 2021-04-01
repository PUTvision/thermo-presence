import copy
import json
import logging
import os
from typing import List
from typing import Tuple
from matplotlib import pyplot as plt
import enum

import cv2
import numpy as np
import pymsgbox

import config
from data_labeling import labeling_config


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


def get_extrapolated_ir_frame_heatmap_flipped(frame_2d, multiplier, interpolation, min_temp, max_temp, colormap):
    new_size = (frame_2d.shape[1] * multiplier, frame_2d.shape[0] * multiplier)
    frame_resized_not_clipped = cv2.resize(src=frame_2d, dsize=new_size, interpolation=interpolation)

    if min_temp is None:
        min_temp = min(frame_resized_not_clipped.reshape(-1))
    if max_temp is None:
        max_temp = max(frame_resized_not_clipped.reshape(-1))
    # logger.debug(f"raw_frame_to_frame_for_video: min={min_temp}, max={max_temp}")

    frame_resized = np.clip(frame_resized_not_clipped, min_temp, max_temp)
    frame_resized_normalized = (frame_resized - min_temp) * (255 / (max_temp - min_temp))
    frame_resized_normalized_u8 = frame_resized_normalized.astype(np.uint8)
    heatmap_u8 = (colormap(frame_resized_normalized_u8) * 2 ** 8).astype(np.uint8)[:, :, :3]
    heatmap_u8_bgr = cv2.cvtColor(heatmap_u8, cv2.COLOR_RGB2BGR)
    heatmap_u8_bgr_flipped = cv2.flip(heatmap_u8_bgr, 1)  # horizontal flip
    return heatmap_u8_bgr_flipped


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


class FrameAnnotation:
    def __init__(self):
        self.accepted = False  # whether frame was marked as annotated successfully
        self.discarded = False  # whether frame was marked as discarded (ignored)
        self.centre_points = []  # type: List[tuple]  # x, y
        self.rectangles = []  # type: List[Tuple[tuple, tuple]]  # (x_left, y_top), (x_right, y_bottom)

        self.raw_frame_data = None  # Not an annotation, but write it to the result file, just in case

    def as_dict(self):
        data_dict = copy.copy(self.__dict__)
        data_dict['centre_points'] = []
        data_dict['rectangles'] = []

        for i, point in enumerate(self.centre_points):
            data_dict['centre_points'].append(xy_on_interpolated_image_to_raw_xy(point))
        for i, rectangle in enumerate(self.rectangles):
            data_dict['rectangles'].append((xy_on_interpolated_image_to_raw_xy(rectangle[0]),
                                            xy_on_interpolated_image_to_raw_xy(rectangle[1])))
        return data_dict

    @classmethod
    def from_dict(cls, data_dict):
        item = cls()
        item.__dict__.update(data_dict)
        for i, point in enumerate(item.centre_points):
            item.centre_points[i] = xy_on_raw_image_to_xy_on_interpolated_image(point)
        for i, rectangle in enumerate(item.rectangles):
            item.rectangles[i] = (xy_on_raw_image_to_xy_on_interpolated_image(rectangle[0]),
                                  xy_on_raw_image_to_xy_on_interpolated_image(rectangle[1]))
        return item


class KeyAction(enum.Enum):
    UNKNOWN = None

    a_ACCEPT = ord('a')
    d_DISCARD = ord('d')
    e_PREVIOUS = ord('e')
    r_NEXT = ord('r')
    c_CLEAR = ord('c')
    z_RECTANGLE = ord('z')
    x_POINT = ord('x')
    q_QUIT = ord('q')

    @classmethod
    def from_pressed_key(cls, key):
        try:
            return cls(key)
        except:
            return KeyAction.UNKNOWN

    def as_char(self):
        return chr(self.value)


class DrawingMode(enum.Enum):
    RECTANGLE = KeyAction.z_RECTANGLE.value
    SINGLE_POINT = KeyAction.x_POINT.value


class SingleFrameAnnotator:
    def __init__(self, ir_frame, rgb_frame, drawing_mode: DrawingMode, initial_annotations: FrameAnnotation = None):
        self.ir_frame_interpolated = get_extrapolated_ir_frame_heatmap_flipped(
            frame_2d=ir_frame,
            multiplier=labeling_config.IR_FRAME_RESIZE_MULTIPLIER,
            interpolation=labeling_config.IR_FRAME_INTERPOLATION_METHOD,
            min_temp=labeling_config.MIN_TEMPERATURE_ON_PLOT,
            max_temp=labeling_config.MAX_TEMPERATURE_ON_PLOT,
            colormap=plt.get_cmap('inferno'))

        self.ir_frame_pixel_resized = get_extrapolated_ir_frame_heatmap_flipped(
            frame_2d=ir_frame,
            multiplier=labeling_config.IR_FRAME_RESIZE_MULTIPLIER,
            interpolation=cv2.INTER_NEAREST_EXACT,
            min_temp=None,
            max_temp=None,
            colormap=plt.get_cmap('viridis'))

        new_rgb_frame_size = (labeling_config.RGB_FRAME_RESIZE_MULTIPLIER * rgb_frame.shape[1],
                              labeling_config.RGB_FRAME_RESIZE_MULTIPLIER * rgb_frame.shape[0])
        self.rgb_frame_resized = cv2.resize(src=rgb_frame, dsize=new_rgb_frame_size, interpolation=cv2.INTER_LINEAR)
        self.initial_annotations = initial_annotations
        self.new_annotations = copy.deepcopy(self.initial_annotations or FrameAnnotation())

        self._button_press_location = None
        self._last_mouse_location = None
        self.drawing_mode = drawing_mode

    def _draw_frame(self):
        ir_frame_interpolated_with_annotation = copy.copy(self.ir_frame_interpolated)
        ir_frame_pixel_resized_with_annotation = copy.copy(self.ir_frame_pixel_resized)

        self.add_annotations_on_img(ir_frame_interpolated_with_annotation)
        self.add_annotations_on_img(ir_frame_pixel_resized_with_annotation)

        cv2.imshow('rgb_frame', self.rgb_frame_resized)
        cv2.imshow('ir_frame_interpolated', ir_frame_interpolated_with_annotation)
        cv2.imshow('ir_frame_pixel_resized', ir_frame_pixel_resized_with_annotation)

        cv2.moveWindow("rgb_frame", 10, 10)
        cv2.moveWindow("ir_frame_interpolated", 610, 10)
        cv2.moveWindow("ir_frame_pixel_resized", 610, 470)

    def get_annotation_for_frame(self) -> Tuple[KeyAction, FrameAnnotation]:
        self._draw_frame()
        cv2.setMouseCallback('ir_frame_interpolated', self._mouse_event)
        cv2.setMouseCallback('ir_frame_pixel_resized', self._mouse_event)

        key_action = KeyAction.UNKNOWN
        while key_action not in [KeyAction.a_ACCEPT, KeyAction.d_DISCARD, KeyAction.e_PREVIOUS,
                          KeyAction.r_NEXT, KeyAction.q_QUIT]:

            key = cv2.waitKey(0)
            key_action = KeyAction.from_pressed_key(key)
            logging.info(f"Key pressed action: {key_action.name}")

            if key_action == KeyAction.x_POINT:
                self.drawing_mode = DrawingMode.SINGLE_POINT
            if key_action == KeyAction.z_RECTANGLE:
                self.drawing_mode = DrawingMode.RECTANGLE

            if key_action == KeyAction.c_CLEAR:
                self.new_annotations = FrameAnnotation()
                self._draw_frame()

        if key_action == KeyAction.a_ACCEPT:
            self.new_annotations.accepted = True
            self.new_annotations.discarded = False
            key_action = KeyAction.r_NEXT
        if key_action == KeyAction.d_DISCARD:
            self.new_annotations.discarded = True
            self.new_annotations.accepted = False
            key_action = KeyAction.r_NEXT

        return key_action, self.new_annotations

    def _mouse_event(self, event, x, y, flags, param):
        redraw = False

        self._last_mouse_location = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self._button_press_location = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._button_press_location is not None:
                redraw = True
        elif event == cv2.EVENT_LBUTTONUP:
            if self._button_press_location:
                self._add_annotation(button_release_location=(x, y))
            self._button_press_location = None
            redraw = True

        if redraw:
            self._draw_frame()

    def _add_annotation(self, button_release_location):
        drawing_mode = self.drawing_mode
        button_press_location = self._button_press_location
        logging.info(f"Adding annotation at {button_press_location}, mode {drawing_mode.name}")

        if drawing_mode == DrawingMode.SINGLE_POINT:
            self.new_annotations.centre_points.append(button_press_location)
        elif drawing_mode == DrawingMode.RECTANGLE:
            self.new_annotations.rectangles.append((button_press_location, button_release_location))

    def add_annotations_on_img(self, img):
        if self.new_annotations.discarded:
            color = (33, 33, 33)
        elif self.new_annotations.accepted:
            color = (0, 120, 60)
        else:
            color = (200, 100, 200)

        for centre_point in self.new_annotations.centre_points:
            cv2.circle(img=img, center=centre_point, color=color, radius=4, thickness=3)
        for top_left, bottom_right in self.new_annotations.rectangles:
            cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=color, thickness=2)

        # draw also annotation in progress
        if self.drawing_mode == DrawingMode.RECTANGLE and self._button_press_location:
            cv2.rectangle(img=img,
                          pt1=self._button_press_location, pt2=self._last_mouse_location,
                          color=color, thickness=3)

        if self.new_annotations.discarded:
            cv2.line(img=img, pt1=(0,0), pt2=img.shape[:2][::-1], color=color, thickness=3)
            cv2.line(img=img, pt1=(img.shape[1], 0), pt2=(0, img.shape[0]), color=color, thickness=3)


class AnnotationCollector:
    ANNOTATIONS_BETWEEN_AUTOSAVE = 1

    def __init__(self, output_file_path, data_batch_dir_path):
        self._output_file_path = output_file_path
        self._annotations = {}  # ir_frame_index: FrameAnnotation
        self._data_batch_dir_path = data_batch_dir_path
        self._annotations_to_autosave = self.ANNOTATIONS_BETWEEN_AUTOSAVE

    def get_annotation(self, ir_frame_index):
        return self._annotations.get(ir_frame_index, FrameAnnotation())

    def set_annotation(self, ir_frame_index, annotation):
        self._annotations[ir_frame_index] = annotation
        if annotation.accepted or annotation.discarded:
            self._annotations_to_autosave -= 1
            if self._annotations_to_autosave == 0:
                self._annotations_to_autosave = self.ANNOTATIONS_BETWEEN_AUTOSAVE
                self.save()

    def save(self):
        data_dict = {
            'output_file_path': self._output_file_path,
            'data_batch_dir_path': self._data_batch_dir_path,
            'annotations': {index: annotation.as_dict() for index, annotation in self._annotations.items()}
        }
        with open(self._output_file_path, 'w') as file:
            file.write(json.dumps(data_dict, indent=2))

    @classmethod
    def load_from_file(cls, file_path):
        item = cls(output_file_path=file_path, data_batch_dir_path=None)
        with open(file_path, 'r') as file:
            data = file.read()
        data_dict = json.loads(data)
        item._data_batch_dir_path = data_dict['data_batch_dir_path']
        item._annotations = {int(index): FrameAnnotation.from_dict(annotation_dict) for index, annotation_dict
                             in data_dict['annotations'].items()}
        return item


def run_labeling_for_data_batch(data_batch_dir_path, output_file_path):
    ir_csv_path = os.path.join(data_batch_dir_path, 'ir.csv')
    rgb_video_path = os.path.join(data_batch_dir_path, 'rgb.mjpeg')

    ir_data_csv_reader = IrDataCsvReader(ir_csv_path)
    rgv_video_reader = RgbVideoReader(rgb_video_path)

    number_of_ir_frames = ir_data_csv_reader.get_number_of_frames()
    number_of_rgb_frames = rgv_video_reader.get_number_of_frames()
    logging.info(f'Number of IR frames: {number_of_ir_frames}. '
                 f'Number of RGB frames: {number_of_rgb_frames}')

    rgb_to_ir_frames_ratio = number_of_rgb_frames / number_of_ir_frames

    drawing_mode = DrawingMode.SINGLE_POINT

    if os.path.isfile(output_file_path):
        logging.info(f"Found a file with annotations ('{output_file_path}') - loading it!")
        try:
            annotation_collector = AnnotationCollector.load_from_file(output_file_path)
        except Exception as e:
            raise Exception(f"Failed to read file '{output_file_path}' with existing annotations!\n"
                            f"Remove this file and retry") from e
    else:
        logging.info(f"A new annotations file will be created '{output_file_path}'")
        annotation_collector = AnnotationCollector(
            output_file_path=output_file_path, data_batch_dir_path=data_batch_dir_path)

    ir_frame_index = 0
    while ir_frame_index < number_of_ir_frames:
        rgb_frame_index = int(ir_frame_index * rgb_to_ir_frames_ratio)
        ir_frame = ir_data_csv_reader.get_frame(ir_frame_index)
        rgb_frame = rgv_video_reader.get_frame(rgb_frame_index)

        msg = f'\nAnnotating frame {ir_frame_index} / {number_of_ir_frames} . Key controls:\n' \
              f'  {KeyAction.a_ACCEPT.as_char()}) accept annotation and move to the next one\n' \
              f'  {KeyAction.d_DISCARD.as_char()}) discard annotation on this frame and move to the next one this frame will be marked as ignored)\n' \
              f'  {KeyAction.e_PREVIOUS.as_char()}) just move to to the previous frame\n' \
              f'  {KeyAction.r_NEXT.as_char()}) just move to to the next frame\n' \
              f'  {KeyAction.c_CLEAR.as_char()}) clear annotation on this frame\n' \
              f'  {KeyAction.z_RECTANGLE.as_char()}) annotate person with a rectangle\n' \
              f'  {KeyAction.x_POINT.as_char()}) annotate person centre with single point\n' \
              f'  {KeyAction.q_QUIT.as_char()}) quit annotating'
        print(msg)

        frame_annotator = SingleFrameAnnotator(
            ir_frame=ir_frame, rgb_frame=rgb_frame,
            drawing_mode=drawing_mode,
            initial_annotations=annotation_collector.get_annotation(ir_frame_index))
        key_action, new_annotiation = frame_annotator.get_annotation_for_frame()
        drawing_mode = frame_annotator.drawing_mode

        # new_annotiation.raw_frame_data = ir_data_csv_reader.get_raw_frame_data(ir_frame_index)  # maybe add raw_data in the file with labels?

        annotation_collector.set_annotation(ir_frame_index, new_annotiation)

        if key_action == KeyAction.r_NEXT:
            ir_frame_index += 1
        if key_action == KeyAction.e_PREVIOUS:
            ir_frame_index = max(0, ir_frame_index - 1)
        if key_action == KeyAction.q_QUIT:
            break

    annotation_collector.save()


def main():
    for i, data_batch_subdir_name in enumerate(labeling_config.SUBDIRECTORIES_TO_ANNOTATE):
        logging.info(f"{i}) Processing data batch '{data_batch_subdir_name}'")
        data_batch_dir_path = os.path.join(labeling_config.ROOT_DATA_DIR_PATH, data_batch_subdir_name)
        output_file_with_labels_name = data_batch_subdir_name.replace('/', '--') + '.csv'
        output_file_with_labels_path = os.path.join(labeling_config.OUTPUT_LABELS_DIR, output_file_with_labels_name)
        if os.path.exists(output_file_with_labels_path):
            msg = f"Looks like the data batch from directory '{data_batch_subdir_name}' is already labeled!" \
                  f"(Output file '{output_file_with_labels_path}' already exists."
            result = pymsgbox.confirm(msg + "\n\nDo you want to edit it anyway?")
            if result != 'OK':
                logging.warning("Skipping data batch!")
                continue

        run_labeling_for_data_batch(data_batch_dir_path=data_batch_dir_path,
                                    output_file_path=output_file_with_labels_path)
        pymsgbox.alert(f"Video '{data_batch_subdir_name}' annotated!")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S')
    main()
    # next/previous frame. Mark point/rectange. Clear. Ignore frame
    # mark center points and rectangeles for each person [remove background?]
    # save to csv (remember about flipped horizontally and scalled!
