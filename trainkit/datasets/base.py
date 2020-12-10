from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Union

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import numpy as np


class BaseDataset(Dataset, ABC):
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):
        super().__init__(**_ignored)

        self.batch_size = hyper_params['batch_size']
        self.num_workers = run_params['num_workers']

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def get_dataloader(self, is_val: bool) -> DataLoader:
        """Initializes and returns DataLoader instance based on self Dataset

        Args:
            is_val: if True than DataLoader returns shuffled batches

        Returns:
            DataLoader instance
        """
        shuffle = False if is_val else True

        loader = DataLoader(dataset=self,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            num_workers=self.num_workers,
                            persistent_workers=True)

        return loader

    @staticmethod
    def calc_class_weights(targets: Union['np.ndarray', torch.Tensor]) -> torch.Tensor:
        """Calculates class weights to balance train dataset

        Args:
            targets: vector with targets, where `0 <= target[i] < num_classes`

        Returns:
            Vector of weights
        """
        count_dict = Counter(targets)
        counts = [count_dict[key] for key in range(len(count_dict))]
        counts = torch.tensor(counts, dtype=torch.float32)

        weights = counts.max() / counts

        return weights
