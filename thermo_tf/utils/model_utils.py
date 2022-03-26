import time
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from data_generator.thermal_data_generator import ThermalDataset
from metrics import CountAccuracy, CountMAE, CountMSE, CountMeanRelativeAbsoluteError


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


def tf_keras_inference(model: tf.keras.Model, inputs: np.ndarray) -> np.ndarray:
    inference_start = time.time()
    outputs = model.predict(inputs)
    inference_time = time.time() - inference_start

    return outputs, inference_time


def tflite_inference(interpreter: Union[tf.lite.Interpreter, str], inputs: np.ndarray) -> np.ndarray:
    if type(interpreter) == str:
        interpreter = tf.lite.Interpreter(interpreter)
    # TFLite allocate tensors.
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale = input_details[0]['quantization'][0]
    in_zero_point = input_details[0]['quantization'][1]
    in_dtype = input_details[0]['dtype']

    out_scale = output_details[0]['quantization'][0]
    out_zero_point = output_details[0]['quantization'][1]

    outputs = []
    inference_time = 0

    for input_data in inputs:
        if (in_scale, in_zero_point) != (0.0, 0):
            input_data = input_data / in_scale + in_zero_point

        interpreter.set_tensor(
            input_details[0]['index'], [input_data.astype(in_dtype)])

        inference_start = time.time()
        interpreter.invoke()
        inference_time += time.time() - inference_start

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if (out_scale, out_zero_point) != (0.0, 0):
            output_data = (output_data - out_zero_point) * out_scale

        outputs.append(output_data)

    return np.vstack(outputs), inference_time


def evaluate(
    model_path: str,
    model_type: str,
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
    mrae_sum = 0

    vec_real_number_of_persons = []
    vec_predicted_number_of_persons = []

    if model_type == 'keras':
        inference_func = tf_keras_inference
        custom_objects = {
            "CountAccuracy": CountAccuracy,
            "CountMAE": CountMAE,
            "CountMSE": CountMSE,
            "CountMeanRelativeAbsoluteError": CountMeanRelativeAbsoluteError
        }
        model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects)
    elif model_type == 'tflite':
        inference_func = tflite_inference
        model = tf.lite.Interpreter(model_path)

    inference_time = 0
    for frames, labels in loader:
        outputs, batch_inference_time = inference_func(model, frames)
        inference_time += batch_inference_time

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
            if rounded_error != 0:
                mrae_sum += (rounded_error + 1) / (true_label + 1) if true_label == 0 else rounded_error / true_label

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
    mrae = mrae_sum / tested_frames

    model_accuracy = correct_count / tested_frames
    model_f1_score = f1_score(vec_real_number_of_persons, np.round(
        vec_predicted_number_of_persons).astype(int), average='weighted')

    average_inference_time = inference_time / tested_frames
    average_fps = 1 / average_inference_time

    print(f"Number of tested frames: {tested_frames}")
    print(f"Average inference time: {round(average_inference_time, 6)}")
    print(f"Average FPS: {round(average_fps,4)}")
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
    print(f'mrae: {mrae}')

    return model_accuracy, model_f1_score, confusion_matrix
