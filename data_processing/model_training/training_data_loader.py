import copy
import os
from pprint import pprint
from typing import List
import numpy as np

import config
from data_labeling.annotation_collector import AnnotationCollector
from data_labeling.ir_data_scv_reader import IrDataCsvReader
from data_labeling.labeling_config import ROOT_DATA_DIR_PATH, OUTPUT_LABELS_DIR
from data_labeling import labeling_config

LABELED_BATCH_DIRS_1 = [
    '31_03_21__318__3or4_people/1/006__11_44_59',
    '31_03_21__318__3or4_people/1/007__11_48_59',
    '31_03_21__318__3or4_people/1/008__11_52_59',
    '31_03_21__318__3or4_people/1/009__11_57_00',
     ]

LABELED_BATCH_DIRS_2 = [
    '31_03_21__318__3or4_people/2/000__14_15_19',
    '31_03_21__318__3or4_people/2/001__14_19_19',
    '31_03_21__318__3or4_people/2/002__14_23_19',
    '31_03_21__318__3or4_people/2/003__14_27_20',
    '31_03_21__318__3or4_people/2/004__14_31_20',
    '31_03_21__318__3or4_people/2/005__14_35_20',
    '31_03_21__318__3or4_people/2/006__14_39_20',
    '31_03_21__318__3or4_people/2/007__14_43_20',
    '31_03_21__318__3or4_people/2/008__14_47_20',
    '31_03_21__318__3or4_people/2/009__14_51_20',
    '31_03_21__318__3or4_people/2/010__14_55_20',
    '31_03_21__318__3or4_people/2/011__14_59_20',
    '31_03_21__318__3or4_people/2/012__15_03_21',
    '31_03_21__318__3or4_people/2/013__15_07_21',
    '31_03_21__318__3or4_people/2/014__15_11_21',
]


class BatchTrainingData:
    """
    Stores training data for one batch
    """
    def __init__(self, min_temperature=20, max_temperature=35):
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
                x_flipped = config.IR_CAMERA_RESOLUTION_X - self.centre_points[i][j][0]
                self.centre_points[i][j] = (x_flipped, self.centre_points[i][j][1])

    def flip_vertically(self):
        for i in range(len(self.raw_ir_data)):
            self.raw_ir_data[i] = np.flip(self.raw_ir_data[i], 0)
            self.normalized_ir_data[i] = np.flip(self.normalized_ir_data[i], 0)
            for j in range(len(self.centre_points[i])):
                y_flipped = config.IR_CAMERA_RESOLUTION_Y - self.centre_points[i][j][1]
                self.centre_points[i][j] = (self.centre_points[i][j][0], y_flipped)


class AugmentedBatchesTrainingData:
    """
    Stores training data for all batches, with data augmentation
    """
    def __init__(self):
        self.batches = []  # Type: List[BatchTrainingData]

    def add_training_batch(self, batch: BatchTrainingData):
        self.batches.append(copy.deepcopy(batch))  # plain data

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


def load_data_for_labeled_batches(labeled_batch_dirs) -> BatchTrainingData:
    labeling_config.IR_FRAME_RESIZE_MULTIPLIER = 1

    training_data = BatchTrainingData()
    for batch_subdir in labeled_batch_dirs:
        data_batch_dir_path = os.path.join(ROOT_DATA_DIR_PATH, batch_subdir)
        raw_ir_data_csv_file_path = os.path.join(data_batch_dir_path, 'ir.csv')
        output_file_with_labels_name = batch_subdir.replace('/', '--') + '.csv'
        annotation_data_file_path = os.path.join(OUTPUT_LABELS_DIR, output_file_with_labels_name)

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


def main():
    training_data = load_data_for_labeled_batches(labeled_batch_dirs=LABELED_BATCH_DIRS_2)
    pprint(training_data.centre_points)
    number_of_persons_on_each_frame = [len(fc) for fc in training_data.centre_points]
    average_number_of_persons = sum(number_of_persons_on_each_frame) / len(number_of_persons_on_each_frame)
    print(f"Number of annotate frames: {len(training_data.centre_points)}, "
          f"with average number of persons on a frame = {average_number_of_persons:.4f}")
    print(f'Number of persons on each frame: {number_of_persons_on_each_frame}')


if __name__ == '__main__':
    main()
