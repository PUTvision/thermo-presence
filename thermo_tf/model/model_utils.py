from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from utils import ThermalDataset


def check_model_prediction(model: tf.keras.Model, config: dict) -> Tuple[float, np.ndarray]:
    raw_frame = np.array(config['dataset']["raw_frame"])

    frame_2d = np.reshape(raw_frame, config['dataset']["IR_camera_resolution"])
    frame_normalized = (frame_2d - config['dataset']["temperature_normalization_min"]) * \
        (1 / (config['dataset']["temperature_normalization_max"] -
         config['dataset']["temperature_normalization_min"]))
    input_frame = np.expand_dims(frame_normalized, axis=(0, 3))

    output_frame = model.predict(input_frame)

    count = np.sum(output_frame) / \
        config['dataset']["sum_of_values_for_one_person"]

    return count, output_frame


def validate_model_with_real_number_of_persons(
        model: tf.keras.Model, 
        loader: ThermalDataset, 
        config: dict, 
        skip_confusion_matrix: bool = False,
    ) -> Tuple[float, float, np.ndarray]:
    """ Validate the model on data from the loader, calculate and print the results and metrics """
    correct_count = 0
    tested_frames = 0
    number_of_frames_with_n_persons = {}
    number_of_frames_with_n_persons_predicted_correctly = {}

    confusion_matrix = np.zeros(shape=(
        config['dataset']["max_people_count"]+1, config['dataset']["max_people_count"]+1), dtype=int)

    mae_sum = 0
    mse_sum = 0

    mae_rounded_sum = 0
    mse_rounded_sum = 0

    vec_real_number_of_persons = []
    vec_predicted_number_of_persons = []

    for frame, labels in loader:
        outputs = model.predict(frame)

        for i in range(len(labels)):
            predicted_img = np.array(outputs[i])
            pred_people = np.sum(predicted_img) / \
                config['dataset']["sum_of_values_for_one_person"]
            pred_label = round(pred_people)

            true_label = round(
                np.sum(labels[i]) / config['dataset']["sum_of_values_for_one_person"])

            if not skip_confusion_matrix:
                confusion_matrix[true_label][pred_label] += 1

            error = abs(pred_people - true_label)
            mae_sum += error
            mse_sum += error*error

            rounded_error = abs(pred_label - true_label)
            mae_rounded_sum += rounded_error
            mse_rounded_sum += rounded_error*rounded_error

            number_of_frames_with_n_persons[pred_label] = \
                number_of_frames_with_n_persons.get(pred_label, 0) + 1

            if true_label == pred_label:
                correct_count += 1
                number_of_frames_with_n_persons_predicted_correctly[pred_label] = \
                    number_of_frames_with_n_persons_predicted_correctly.get(
                        pred_label, 0) + 1

            vec_real_number_of_persons.append(true_label)
            vec_predicted_number_of_persons.append(pred_people)
            tested_frames += 1

    mae = mae_sum / tested_frames
    mse = mse_sum / tested_frames
    mae_rounded = mae_rounded_sum / tested_frames
    mse_rounded = mse_rounded_sum / tested_frames

    model_accuracy = correct_count / tested_frames
    model_f1_score = f1_score(vec_real_number_of_persons, np.round(
        vec_predicted_number_of_persons).astype(int), average='weighted')

    print(f"Number of tested frames: {tested_frames}")
    print(f"Model Accuracy = {model_accuracy}")
    print(f"Model F1 score = {model_f1_score}")
    print('Predicted:\n' + '\n'.join([f'   {count} frames with {no} persons' for no,
          count in sorted(number_of_frames_with_n_persons.items())]))
    print('Predicted correctly:\n' + '\n'.join([f'   {count} frames with {no} persons' for no, count in sorted(
        number_of_frames_with_n_persons_predicted_correctly.items())]))
    print(f'mae: {mae}')
    print(f'mse: {mse}')
    print(f'mae_rounded: {mae_rounded}')
    print(f'mse_rounded: {mse_rounded}')

    return model_accuracy, model_f1_score, confusion_matrix
