from typing import Union

import numpy as np


class ImageDatasetProperties:
    _mean: np.ndarray
    _std: np.ndarray
    _min: np.ndarray
    _max: np.ndarray

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value: Union[str, list]):
        """Sets mean attribute

        Args:
            value: a comma separated string or list with 1 or 3 elements
        """
        self._mean = self.__convert_mean_std_input(value, 'mean')

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value: Union[str, list]):
        """Sets std attribute

        Args:
            value: a comma separated string or list with 1 or 3 elements
        """
        self._std = self.__convert_mean_std_input(value, 'std')

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value: Union[int, float]):
        self._min = self.__convert_min_max_input(value, 'min')

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value: Union[int, float]):
        self._max = self.__convert_min_max_input(value, 'max')

    @staticmethod
    def __convert_mean_std_input(value: Union[str, list],
                                 param_name: str) -> np.ndarray:
        value_list = value.split(',') if isinstance(value, str) else value

        if len(value_list) == 1:
            value_array = np.array(value_list[0], dtype=np.float32)
        elif len(value_list) == 3:
            value_array = np.array(value_list, dtype=np.float32).reshape((1, 1, 3))
        else:
            raise ValueError(f'Param "{param_name}" must have 3 elements')

        return value_array

    @staticmethod
    def __convert_min_max_input(value: Union[int, float],
                                param_name: str) -> np.ndarray:
        if isinstance(value, (int, float)):
            value_array = np.array(value, dtype=np.float32)
        else:
            raise ValueError(f'Param "{param_name}" must be int or float')

        return value_array
