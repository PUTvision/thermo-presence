from typing import Tuple

import numpy as np
import tensorflow as tf

from utils.data_utils import AugmentedBatchesTrainingData, get_img_reconstructed_from_labels


class ThermalDataset(tf.keras.utils.Sequence):

    def __init__(self, augmented_data: AugmentedBatchesTrainingData, batch_size:int=16, shuffle:bool=False, task:str='density_estimation'):
        self.augmented_data = AugmentedBatchesTrainingData
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task = task
        self._index_to_batch_and_subindex_map = {}
        
        self._cache = {}
        
        i = 0
        for batch in augmented_data.batches:
            for j in range(len(batch.raw_ir_data)):
                self._index_to_batch_and_subindex_map[i] = (batch, j) 
                i += 1

    def __len__(self) -> int:
        return len(self._index_to_batch_and_subindex_map) // self.batch_size

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if batch_idx not in self._cache:
            frames = []
            reconstructions = []
            number_of_people = []
            for i in range(self.batch_size):
                idx = batch_idx * self.batch_size + i
                batch, subindex = self._index_to_batch_and_subindex_map[idx]
                frame = batch.normalized_ir_data[subindex][np.newaxis, :, :][..., np.newaxis]
                frames.append(frame)

                batch, subindex = self._index_to_batch_and_subindex_map[idx]
                centre_points = batch.centre_points[subindex]
                img_reconstructed = get_img_reconstructed_from_labels(centre_points)
                img_reconstructed_3d = img_reconstructed[np.newaxis, :, :]
                reconstructions.append(img_reconstructed_3d)

                nop = self.get_number_of_persons_for_frame(batch_idx*self.batch_size + i)
                nop_one_hot = np.zeros(6)
                nop_one_hot[nop] = 1
                number_of_people.append(nop_one_hot.astype(int))

            if self.task == 'classification':
                result = np.vstack(frames), np.vstack(number_of_people)
            else:
                result = np.vstack(frames), np.vstack(reconstructions)

            self._cache[batch_idx] = result
            
        return self._cache[batch_idx]
    
    def get_number_of_persons_for_frame(self, idx):
        batch, subindex = self._index_to_batch_and_subindex_map[idx]
        return len(batch.centre_points[subindex])