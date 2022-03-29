import json
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import tensorflow as tf
from albumentations import Compose, Flip, Affine, KeypointParams, RandomGridShuffle
from scipy.ndimage import gaussian_filter


class ThermalDataset(tf.keras.utils.Sequence):
    def __init__(self, data_path: Path, sequences_names: Sequence[str], person_point_weight: float,
                 batch_size: int, augment: bool = False, task: str = 'density_estimation'):
        self._frames = self._load_frames(data_path, sequences_names)
        self._labels = self._load_labels(data_path, sequences_names)
        self._person_point_weight = person_point_weight
        self._batch_size = batch_size
        self.task = task
        self._augment = augment
        self._augmentations = Compose([
            Affine(scale=(0.9, 1.1), rotate=(-15, 15), shear=(-5, 5), translate_percent=(-10, 10)),
            Flip()
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))
        self._transforms = Compose([], keypoint_params=KeypointParams(format='xy', remove_invisible=True))

    @staticmethod
    def _load_frames(data_path: Path, sequences_names: Sequence[str]) -> List[np.ndarray]:
        frames = []
        for sequence_name in sequences_names:
            with (data_path / 'data' / f'{sequence_name}.json').open() as file:
                sequence_data = json.load(file)

            for frame_data in sequence_data:
                frame = np.asarray(frame_data['data'], dtype=np.float32)
                frame -= 20
                frame /= 15
                frames.append(frame)

        return frames

    @staticmethod
    def _load_labels(data_path: Path, sequences_names: Sequence[str]) -> List[List[List[float]]]:
        labels = []
        for sequence_name in sequences_names:
            with (data_path / 'labels' / f'{sequence_name}.json').open() as file:
                sequence_labels = json.load(file)

            labels.extend(sequence_labels)

        return labels

    @staticmethod
    def generate_mask(keypoints: List[Tuple[int, int]], image_shape: Tuple[int, int], person_point_weight: float, sigma: Tuple[int, int] = (3,3)):
        label = np.zeros(image_shape, dtype=np.float32)

        for key in keypoints:
            x, y = map(int, key)
            label[y, x] = person_point_weight

        label = gaussian_filter(label, sigma=sigma, order=0)

        return np.array([label])

    def __len__(self):
        return len(self._frames) // self._batch_size

    def __getitem__(self, batch_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        frames = []
        masks = []
        number_of_people = []
        for i in range(self._batch_size):
            idx = batch_idx * self._batch_size + i

            frame, keypoints = self._frames[idx], self._labels[idx]
            for keypoint in keypoints:
                keypoint[0] = min(keypoint[0], frame.shape[1] - 1)
                keypoint[1] = min(keypoint[1], frame.shape[0] - 1)

            if self._augment:
                transformed = self._augmentations(image=frame, keypoints=keypoints)
            else:
                transformed = self._transforms(image=frame, keypoints=keypoints)

            frame, keypoints = transformed['image'], transformed['keypoints']
            mask = self.generate_mask(keypoints, frame.shape, self._person_point_weight)

            nop = len(keypoints)
            nop_one_hot = np.zeros(6)
            nop_one_hot[nop] = 1

            frames.append(frame)
            masks.append(mask)
            number_of_people.append(nop_one_hot.astype(int))

        if self.task == 'classification':
            result = np.expand_dims(np.stack(frames), axis=-1), np.vstack(number_of_people)
        else:
            result = np.expand_dims(np.stack(frames), axis=-1), np.vstack(masks)

        return result
