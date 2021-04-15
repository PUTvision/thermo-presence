import logging
import os

import pymsgbox

from data_labeling.annotation_collector import AnnotationCollector
from data_labeling import labeling_config
from data_labeling.ir_data_scv_reader import IrDataCsvReader
from data_labeling.rgb_video_reader import RgbVideoReader
from data_labeling.single_frame_annotator import DrawingMode, KeyAction, SingleFrameAnnotator


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
    automove_to_next_frame_after_mouse_released = False

    annotation_collector = _create_annotation_collector(output_file_path=output_file_path,
                                                        data_batch_dir_path=data_batch_dir_path)

    ir_frame_index = 0
    quit_application = False
    while ir_frame_index < number_of_ir_frames:
        rgb_frame_index = int(ir_frame_index * rgb_to_ir_frames_ratio)
        ir_frame = ir_data_csv_reader.get_frame(ir_frame_index)
        rgb_frame = rgv_video_reader.get_frame(rgb_frame_index)

        help_msg = f'\nAnnotating frame {ir_frame_index} / {number_of_ir_frames}. ' \
                   f'{KeyAction.get_help_message()}'
        print(help_msg)

        frame_annotator = SingleFrameAnnotator(
            ir_frame=ir_frame,
            rgb_frame=rgb_frame,
            drawing_mode=drawing_mode,
            automove_to_next_frame_after_mouse_released=automove_to_next_frame_after_mouse_released,
            initial_annotations=annotation_collector.get_annotation(ir_frame_index))
        key_action, new_annotation = frame_annotator.get_annotation_for_frame()
        drawing_mode = frame_annotator.drawing_mode
        automove_to_next_frame_after_mouse_released = frame_annotator.automove_to_next_frame_after_mouse_released

        # maybe add raw_data in the file with labels?
        # new_annotiation.raw_frame_data = ir_data_csv_reader.get_raw_frame_data(ir_frame_index)

        annotation_collector.set_annotation(ir_frame_index, new_annotation)

        if key_action == KeyAction.r_NEXT:
            ir_frame_index += 1
        if key_action == KeyAction.e_PREVIOUS:
            ir_frame_index = max(0, ir_frame_index - 1)
        if key_action == KeyAction.q_QUIT:
            quit_application = True
            break

    annotation_collector.save()
    return quit_application


def _create_annotation_collector(output_file_path, data_batch_dir_path) -> AnnotationCollector:
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

    return annotation_collector


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

        if run_labeling_for_data_batch(data_batch_dir_path=data_batch_dir_path,
                                       output_file_path=output_file_with_labels_path):
            break

        pymsgbox.alert(f"Video '{data_batch_subdir_name}' annotated!")

    logging.info("Annotatingq finished or closed")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S')
    main()
