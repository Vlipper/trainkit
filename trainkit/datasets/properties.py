from typing import Union

import numpy as np


class ImageDatasetProperties:
    _mean: np.ndarray
    _std: np.ndarray

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value: str):
        """Sets mean attribute

        Args:
            value: a comma separated string or list with 1 or 3 elements
        """
        self._mean = self.__convert_input(value, 'mean')

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value: str):
        """Sets std attribute

        Args:
            value: a comma separated string or list with 1 or 3 elements
        """
        self._std = self.__convert_input(value, 'std')

    @staticmethod
    def __convert_input(value: Union[str, list],
                        param_name: str) -> np.ndarray:
        value_list = value.split(',') if isinstance(value, str) else value

        if len(value_list) == 1:
            value_array = np.array(value_list[0], dtype=np.float32)
        elif len(value_list) == 3:
            value_array = np.array(value_list, dtype=np.float32).reshape((1, 1, 3))
        else:
            raise ValueError(f'Param "{param_name}" must have 3 elements')

        return value_array
