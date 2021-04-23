from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from typing import Union
    import numpy as np


class BaseDataset(Dataset,
                  ABC):
    @abstractmethod
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @staticmethod
    def calc_class_weights(targets: 'Union[np.ndarray, torch.Tensor]') -> torch.Tensor:
        """Calculates class weights to balance train dataset

        Args:
            targets: vector with targets, where `0 <= target[i] < num_classes`

        Returns:
            Vector of weights
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        count_dict = Counter(targets)
        counts = [count_dict[key] for key in range(len(count_dict))]
        counts = torch.tensor(counts, dtype=torch.float32)

        weights = counts.max() / counts

        return weights
