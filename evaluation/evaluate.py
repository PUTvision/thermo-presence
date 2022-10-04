import os
import time
from functools import partial
from typing import Tuple, List

import click
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


TEST_DIRS = ["008__13_26_20", "009__14_51_20", "010__14_55_20", "011__14_59_20", "015__13_54_21"]
SINGLE_BLOB_MEAN_VALUE = 51.35
IMG_SHAPE = [24,32]


def generate_mask(keypoints: List[Tuple[int, int]], sigma: Tuple[int, int] = (3,3)):
    label = np.zeros(IMG_SHAPE, dtype=np.float32)

    for key in keypoints:
        x, y = np.clip(list(map(round, key)), [0,0], IMG_SHAPE)
        label[y, x] = SINGLE_BLOB_MEAN_VALUE

    label = gaussian_filter(label, sigma=sigma, order=0)

    return np.array([label])


def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    test_input_data = []
    test_output_data = []

    for filename in TEST_DIRS:
        df = pd.read_hdf(f'{data_path}/{filename}.h5')
        data = df['data'].to_numpy()
        points = df['points'].to_numpy()

        test_input_data.append(
            np.stack([np.array(data[i], dtype=float) for i in range(len(data))])
        )
        test_output_data.append(
            np.stack([np.array(generate_mask(points[i]), dtype=float) for i in range(len(points))])
        )

    return np.expand_dims(np.vstack(test_input_data), -1), np.vstack(test_output_data).reshape((-1, *IMG_SHAPE, 1))


def tflite_inference(input_data, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale = input_details[0]['quantization'][0]
    in_zero_point = input_details[0]['quantization'][1]
    in_dtype = input_details[0]['dtype']

    out_scale = output_details[0]['quantization'][0]
    out_zero_point = output_details[0]['quantization'][1]

    if (in_scale, in_zero_point) != (0.0, 0):
        input_data = input_data / in_scale + in_zero_point

    interpreter.set_tensor(input_details[0]['index'], [input_data.astype(in_dtype)])

    inference_start = time.time()
    interpreter.invoke()
    inference_time = time.time() - inference_start

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if (out_scale, out_zero_point) != (0.0, 0):
        output_data = (output_data - out_zero_point) * out_scale

    return output_data, inference_time


def myriad_inference(input_data, interpreter, input_blob):
    inference_start = time.time()
    output = interpreter.infer(inputs={input_blob: input_data})
    inference_time = time.time() - inference_start

    output_data = np.array(list(output.values())[0])

    return output_data, inference_time


@click.command()
@click.option('--inference_type', help='Inference framework (device)', type=click.Choice(['tflite', 'edgetpu', 'myriad'], case_sensitive=True))
@click.option('--model_path', help='Path to model', type=str)
@click.option('--data_path', help='Path to HDF files', type=str, default='../dataset/hdfs')
def main(inference_type, model_path, data_path):
    test_input_arr, test_output_arr = load_test_data(data_path)
    print(f'Test array shape: {test_input_arr.shape}')

    inference_full_time = 0.0
    output_count = []
    output_masks = []

    if inference_type == 'tflite':
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path)
        interpreter.allocate_tensors()

        inference_func = partial(tflite_inference, interpreter=interpreter)
    elif inference_type == 'edgetpu':
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path)
        interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        interpreter.allocate_tensors()

        inference_func = partial(tflite_inference, interpreter=interpreter)
    elif inference_type == 'myriad':
        import ngraph as ng
        from openvino.inference_engine import IECore

        ie = IECore()

        net = ie.read_network(model_path, os.path.splitext(model_path)[0] + ".bin")
        input_blob = next(iter(net.input_info))
        interpreter = ie.load_network(network=net, device_name='MYRIAD', num_requests=1)

        inference_func = partial(myriad_inference, interpreter=interpreter, input_blob=input_blob)
    
    for arr_num in tqdm(range(test_input_arr.shape[0])):
        arr = test_input_arr[arr_num]

        test_output_mask, test_infer_time = inference_func(arr)
        test_count = np.sum(test_output_mask) / SINGLE_BLOB_MEAN_VALUE

        inference_full_time += test_infer_time
        output_count.append(test_count)
        output_masks.append(test_output_mask)

    output_masks = np.vstack(output_masks)
    np.save(f'rpi_{inference_type}_output.npy', output_masks)

    input_size = test_output_arr.shape[1] * test_output_arr.shape[2] * test_output_arr.shape[3]
    MSE = np.sum(np.power(test_output_arr - output_masks, 2)) / input_size / test_input_arr.shape[0]
    MAE = np.sum(np.abs(test_output_arr - output_masks)) / input_size / test_input_arr.shape[0]

    count_diff = np.sum(test_output_arr, axis=(1,2,3)) / SINGLE_BLOB_MEAN_VALUE - output_count
    count_MSE = np.sum(np.power(count_diff, 2)) / test_input_arr.shape[0]
    count_MAE = np.sum(np.abs(count_diff)) / test_input_arr.shape[0]

    count_rounded_diff = np.round(np.sum(test_output_arr, axis=(1,2,3)) / SINGLE_BLOB_MEAN_VALUE) - np.round(output_count)
    count_rounded_MSE = np.sum(np.power(count_rounded_diff, 2)) / test_input_arr.shape[0]
    count_rounded_MAE = np.sum(np.abs(count_rounded_diff)) / test_input_arr.shape[0]

    average_infer_time = inference_full_time / test_input_arr.shape[0]
    
    print(f'Mean square error: {MSE}')
    print(f'Mean absolute error: {MAE}')
    print(f'Count mean square error: {count_MSE}')
    print(f'Count mean absolute error: {count_MAE}')
    print(f'Rounded count mean square error: {count_rounded_MSE}')
    print(f'Rounded count mean absolute error: {count_rounded_MAE}')
    print(f'Average inference time: {average_infer_time} ms')


if __name__ == '__main__':
    main()
