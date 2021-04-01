import copy
import enum
import logging
from typing import Tuple

import cv2
from matplotlib import pyplot as plt

from data_labeling import labeling_config
from data_labeling.frame_annotation import FrameAnnotation
from data_labeling.labeling_utils import get_extrapolated_ir_frame_heatmap_flipped


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

    @classmethod
    def get_help_message(cls):
        return f'Key controls:\n' \
               f'  {KeyAction.a_ACCEPT.as_char()}) accept annotation and move to the next one\n' \
               f'  {KeyAction.d_DISCARD.as_char()}) discard annotation on this frame and move to the next one this frame will be marked as ignored)\n' \
               f'  {KeyAction.e_PREVIOUS.as_char()}) just move to to the previous frame\n' \
               f'  {KeyAction.r_NEXT.as_char()}) just move to to the next frame\n' \
               f'  {KeyAction.c_CLEAR.as_char()}) clear annotation on this frame\n' \
               f'  {KeyAction.z_RECTANGLE.as_char()}) annotate person with a rectangle\n' \
               f'  {KeyAction.x_POINT.as_char()}) annotate person centre with single point\n' \
               f'  {KeyAction.q_QUIT.as_char()}) quit annotating'


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
