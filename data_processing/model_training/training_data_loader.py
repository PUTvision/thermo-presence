import os
from pprint import pprint
from typing import List
import numpy as np

from data_labeling.annotation_collector import AnnotationCollector
from data_labeling.ir_data_scv_reader import IrDataCsvReader
from data_labeling.labeling_config import ROOT_DATA_DIR_PATH, OUTPUT_LABELS_DIR
from data_labeling import labeling_config

LABELED_BATCHES_DIRS = [
    '31_03_21__318__3or4_people/1/006__11_44_59',
    '31_03_21__318__3or4_people/1/007__11_48_59',
    '31_03_21__318__3or4_people/1/008__11_52_59',
    #'31_03_21__318__3or4_people/1/009__11_57_00',
    ]


class TrainingData:
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


def load_data_for_labeled_batches(labeled_batch_dirs) -> TrainingData:
    labeling_config.IR_FRAME_RESIZE_MULTIPLIER = 1

    training_data = TrainingData()
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
    training_data = load_data_for_labeled_batches(labeled_batch_dirs=LABELED_BATCHES_DIRS)
    pprint(training_data.centre_points)
    number_of_persons_on_each_frame = [len(fc) for fc in training_data.centre_points]
    average_number_of_persons = sum(number_of_persons_on_each_frame) / len(number_of_persons_on_each_frame)
    print(f"Number of annotate frames: {len(training_data.centre_points)}, "
          f"with average number of persons on a frame = {average_number_of_persons:.4f}")
    print(f'Number of persons on each frame: {number_of_persons_on_each_frame}')

if __name__ == '__main__':
    main()
