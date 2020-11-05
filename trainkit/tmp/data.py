from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class DatasetProperties:
    _mean: Optional[np.ndarray]
    _std: Optional[np.ndarray]

    @staticmethod
    def __str_converter(value: str, param_name: str) -> np.ndarray:
        value_list = value.split(',')
        if len(value_list) != 3:
            raise ValueError('param "{}" must be csv string with 3 elements'.format(param_name))

        value_array = np.array(value_list, dtype=np.float32).reshape((1, 1, 3))

        return value_array

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value: str):
        self._mean = self.__str_converter(value, 'mean')

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value: str):
        self._std = self.__str_converter(value, 'std')


class DatasetBaseMixin(DatasetProperties, Dataset):
    def __init__(self, run_params: dict, hyper_params: dict, **kwargs):
        super().__init__()

        self.desc = self.__read_data_description(kwargs['description_csv_path'])
        # self.sampler = kwargs['sampler']

        self.num_workers = run_params['num_workers']

        self.batch_size = hyper_params['batch_size']
        self.mean, self.std = hyper_params['mean'], hyper_params['std']

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    @staticmethod
    def __read_data_description(desc_path: Path) -> np.ndarray:
        desc = pd.read_csv(desc_path)
        desc = desc.values

        return desc

    def _standartize(self, img: np.ndarray) -> np.ndarray:
        img_mod = (img - self.mean) / self.std

        return img_mod

    @staticmethod
    def _reorder_img_axes(img: np.ndarray, axes_order: Tuple[int, ...]) -> np.ndarray:
        img_mod = np.transpose(img, axes_order)

        return img_mod

    def get_dataloader(self, is_val: bool):
        shuffle = False if is_val else True
        # sampler=self.sampler,
        loader = DataLoader(dataset=self, batch_size=self.batch_size, shuffle=shuffle,
                            num_workers=self.num_workers)

        return loader
