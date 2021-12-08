import os
import json
import copy
from typing import List

import cv2
import numpy as np


# For Gauss function with sig=3, sum of values for one person is between 45 - 55, depending whether people are on the edge. 
# To predict the number of people on a frame as a sum of pixels of the predicted image, we need to divide every pixel by a constant   
sum_of_values_for_one_person = 51.35  # total sum of pixels for one person on the reconstructed image, of course it changes with gaussian function parameters. Calculated as average from all training data

IR_CAMERA_RESOLUTION_X = 32
IR_CAMERA_RESOLUTION_Y = 24

IR_CAMERA_RESOLUTION = (IR_CAMERA_RESOLUTION_Y, IR_CAMERA_RESOLUTION_X)

IR_CAMERA_RESOLUTION_XY = (IR_CAMERA_RESOLUTION_X, IR_CAMERA_RESOLUTION_Y)  # for opencv functions

# for frames normalization
TEMPERATURE_NORMALIZATION__MIN = 20
TEMPERATURE_NORMALIZATION__MAX = 35

IR_FRAME_RESIZE_MULTIPLIER = 1

MIN_TEMPERATURE_ON_PLOT = 20  # None for auto range
MAX_TEMPERATURE_ON_PLOT = 30  # None for auto range
IR_FRAME_INTERPOLATION_METHOD = cv2.INTER_CUBIC
RGB_FRAME_RESIZE_MULTIPLIER = 2


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
            frame_2d = np.reshape(frame_data_1d, IR_CAMERA_RESOLUTION)
            frames.append(frame_2d)
        return frames, raw_frame_data

    def get_number_of_frames(self):
        return len(self._frames)

    def get_frame(self, n):
        return self._frames[n]

    def get_raw_frame_data(self, n) -> str:
        return self._raw_frame_data[n]



class BatchTrainingData:
    """
    Stores training data for one batch
    """
    def __init__(self, min_temperature=TEMPERATURE_NORMALIZATION__MIN, max_temperature=TEMPERATURE_NORMALIZATION__MAX):
        self.centre_points = []  # type: List[List[tuple]]
        self.raw_ir_data = []  # type: List[np.ndarray]
        self.normalized_ir_data = []  # type: List[np.ndarray]  # same data as raw_ir_data, but normalized

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def append_frame_data(self, centre_points, raw_ir_data):
        self.centre_points.append(centre_points)
        self.raw_ir_data.append(raw_ir_data)

        ir_data_normalized = (raw_ir_data - self.min_temperature) * (1 / (self.max_temperature - self.min_temperature))
        self.normalized_ir_data.append(ir_data_normalized)

    def flip_horizontally(self):
        for i in range(len(self.raw_ir_data)):
            self.raw_ir_data[i] = np.flip(self.raw_ir_data[i], 1)
            self.normalized_ir_data[i] = np.flip(self.normalized_ir_data[i], 1)
            for j in range(len(self.centre_points[i])):
                x_flipped = IR_CAMERA_RESOLUTION_X - self.centre_points[i][j][0]
                self.centre_points[i][j] = (x_flipped, self.centre_points[i][j][1])

    def flip_vertically(self):
        for i in range(len(self.raw_ir_data)):
            self.raw_ir_data[i] = np.flip(self.raw_ir_data[i], 0)
            self.normalized_ir_data[i] = np.flip(self.normalized_ir_data[i], 0)
            for j in range(len(self.centre_points[i])):
                y_flipped = IR_CAMERA_RESOLUTION_Y - self.centre_points[i][j][1]
                self.centre_points[i][j] = (self.centre_points[i][j][0], y_flipped)


class AugmentedBatchesTrainingData:
    """
    Stores training data for all batches, with data augmentation
    """
    def __init__(self):
        self.batches = []  # Type: List[BatchTrainingData]

    def add_training_batch(self, batch: BatchTrainingData, flip_and_rotate=True):
        self.batches.append(copy.deepcopy(batch))  # plain data

        if flip_and_rotate:
            batch.flip_horizontally()
            self.batches.append(copy.deepcopy(batch))  # flipped horizontally

            batch.flip_vertically()
            self.batches.append(copy.deepcopy(batch))  # rotated 180 degrees

            batch.flip_horizontally()
            self.batches.append(copy.deepcopy(batch))  # rotated vertically

    def print_stats(self):
        total_number_of_frames = 0
        number_of_frames_with_n_persons = {}
        for batch in self.batches:
            total_number_of_frames += len(batch.centre_points)
            for centre_points in batch.centre_points:
                number_of_persons = len(centre_points)
                number_of_frames_with_n_persons[number_of_persons] = \
                    number_of_frames_with_n_persons.get(number_of_persons, 0) + 1
        frames_persons_details_msg = '\n'.join([f'   {count} frames with {no} persons'
                                               for no, count in number_of_frames_with_n_persons.items()])
        msg = f"AugmentedBatchesTrainingData with {len(self.batches)} BatchTrainingData batches.\n" \
              f"Total number of frames after augmentation: {total_number_of_frames}, with:\n" \
              f"{frames_persons_details_msg}"
        print(msg)


def load_data_for_labeled_batches(labeled_batch_dirs: List[str], project_data_dir: str = './') -> BatchTrainingData:
    data_dir_path = os.path.join(project_data_dir, 'data')
    labels_dir_path = os.path.join(project_data_dir, 'labels')

    training_data = BatchTrainingData()
    for batch_subdir in labeled_batch_dirs:
        data_batch_dir_path = os.path.join(data_dir_path, batch_subdir)
        raw_ir_data_csv_file_path = os.path.join(data_batch_dir_path, 'ir.csv')
        output_file_with_labels_name = batch_subdir.replace('/', '--') + '.csv'
        annotation_data_file_path = os.path.join(labels_dir_path, output_file_with_labels_name)

        raw_ir_data_csv_reader = IrDataCsvReader(file_path=raw_ir_data_csv_file_path)
        annotations_collector = AnnotationCollector.load_from_file(

            file_path=annotation_data_file_path, do_not_scale_and_reverse=True)

        for frame_index in range(raw_ir_data_csv_reader.get_number_of_frames()):
            raw_frame_data = raw_ir_data_csv_reader.get_frame(frame_index)
            frame_annotations = annotations_collector.get_annotation(frame_index)
            if not frame_annotations.accepted:
                print(f"Frame index {frame_index} from batch '{batch_subdir}' not annotated!")
                continue
            training_data.append_frame_data(
                centre_points=frame_annotations.centre_points,
                raw_ir_data=raw_frame_data)

    return training_data


class AnnotationCollector:
    ANNOTATIONS_BETWEEN_AUTOSAVE = 10

    def __init__(self, output_file_path, data_batch_dir_path):
        self._output_file_path = output_file_path
        self._annotations = {}  # ir_frame_index: FrameAnnotation
        self._data_batch_dir_path = data_batch_dir_path
        self._annotations_to_autosave = 1

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
            'annotations': {index: annotation.as_dict()
                            for index, annotation in self._annotations.items()}
        }
        with open(self._output_file_path, 'w') as file:
            file.write(json.dumps(data_dict, indent=2))
            file.flush()

    @classmethod
    def load_from_file(cls, file_path, do_not_scale_and_reverse=False):
        item = cls(output_file_path=file_path, data_batch_dir_path=None)
        with open(file_path, 'r') as file:
            data = file.read()
        data_dict = json.loads(data)
        item._data_batch_dir_path = data_dict['data_batch_dir_path']
        item._annotations = {int(index): FrameAnnotation.from_dict(annotation_dict, do_not_scale_and_reverse) for index, annotation_dict
                             in data_dict['annotations'].items()}
        return item


def x_on_interpolated_image_to_raw_x(x):
    x_raw_flipped = x / IR_FRAME_RESIZE_MULTIPLIER
    x_raw = IR_CAMERA_RESOLUTION_X - x_raw_flipped
    return x_raw


def y_on_interpolated_image_to_raw_y(y):
    return y / IR_FRAME_RESIZE_MULTIPLIER


def xy_on_interpolated_image_to_raw_xy(xy: tuple) -> tuple:
    return (x_on_interpolated_image_to_raw_x(xy[0]),
            y_on_interpolated_image_to_raw_y(xy[1]))


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
    def from_dict(cls, data_dict, do_not_scale_and_reverse=False):
        item = cls()
        item.__dict__.update(data_dict)
        for i, point in enumerate(item.centre_points):
            item.centre_points[i] = xy_on_raw_image_to_xy_on_interpolated_image(point, do_not_scale_and_reverse)
        for i, rectangle in enumerate(item.rectangles):
            item.rectangles[i] = (xy_on_raw_image_to_xy_on_interpolated_image(rectangle[0], do_not_scale_and_reverse),
                                  xy_on_raw_image_to_xy_on_interpolated_image(rectangle[1], do_not_scale_and_reverse))
        return item


def x_on_interpolated_image_to_raw_x(x):
    x_raw_flipped = x / IR_FRAME_RESIZE_MULTIPLIER
    x_raw = IR_CAMERA_RESOLUTION_X - x_raw_flipped
    return x_raw


def y_on_interpolated_image_to_raw_y(y):
    return y / IR_FRAME_RESIZE_MULTIPLIER


def xy_on_interpolated_image_to_raw_xy(xy: tuple) -> tuple:
    return (x_on_interpolated_image_to_raw_x(xy[0]),
            y_on_interpolated_image_to_raw_y(xy[1]))


def x_on_raw_image_to_x_on_interpolated_image(x):
    x_flipped = IR_CAMERA_RESOLUTION_X - x
    return round(x_flipped * IR_FRAME_RESIZE_MULTIPLIER)


def y_on_raw_image_to_y_on_interpolated_image(y):
    return round(y * IR_FRAME_RESIZE_MULTIPLIER)


def xy_on_raw_image_to_xy_on_interpolated_image(xy: tuple, do_not_scale_and_reverse=False) -> tuple:
    if do_not_scale_and_reverse:
        return xy

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


def draw_airbrush_circle(img, centre, radius):
    for x in range(max(0, round(centre[0]-radius)), min(img.shape[0], round(centre[0]+radius+1))):
        for y in range(max(0, round(centre[1]-radius)), min(img.shape[1], round(centre[1]+radius+1))):
            point = (x, y)
            distance_to_centre = cv2.norm((centre[0] - x, centre[1] - y))
            if distance_to_centre > radius:
                continue
            img[point] += 1 - distance_to_centre / radius
            

def draw_cross(img, centre, cross_width, cross_height):
    for x in range(max(0, round(centre[0]) - cross_width), min(img.shape[0], round(centre[0]) + cross_width + 1)):
        for y in range(max(0, round(centre[1]) - cross_height), min(img.shape[1], round(centre[1]) + cross_height + 1)):
            point = (x, y)
            img[point] = 1
    
    for x in range(max(0, round(centre[0] - cross_height)), min(img.shape[0], round(centre[0] + cross_height + 1))):
        for y in range(max(0, round(centre[1] - cross_width)), min(img.shape[1], round(centre[1] + cross_width + 1))):
            point = (x, y)
            img[point] = 1
            

def gauss_1d(x, sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
    
    
def draw_gauss(img, sig, centre):
    radius = 3 * sig
    for x in range(max(0, round(centre[0]-radius)), min(img.shape[0], round(centre[0]+radius+1))):
        for y in range(max(0, round(centre[1]-radius)), min(img.shape[1], round(centre[1]+radius+1))):
            point = (x, y)
            distance_to_centre = cv2.norm((centre[0] - x, centre[1] - y))
            if distance_to_centre > radius:
                continue
            gauss_value = gauss_1d(distance_to_centre, sig)
            img[point] += gauss_value
    
    
def get_img_reconstructed_from_labels(centre_points):
    """ Function used to create training data basing on the labels """
    
    img_reconstructed = np.zeros(shape=(IR_CAMERA_RESOLUTION[0], 
                                        IR_CAMERA_RESOLUTION[1]))

    for centre_point in centre_points:
        centre_point = centre_point[::-1]  # reversed x and y in 
        draw_gauss(img=img_reconstructed, 
                   centre=[c for c in centre_point], 
                   sig=3)
    
    #img_int = (img_reconstructed * (NUMBER_OF_OUPUT_CLASSES-1)).astype('int')
    #return img_int
    return img_reconstructed
