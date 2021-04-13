from abc import ABC
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing import List, Tuple, Union
    from pathlib import Path
    from albumentations import Compose
    from numpy import ndarray


__all__ = [
    'ImageBaseOperationsMixin',
    'ImageScalingMixin',
    'ImageMixingMixin',
    'ImageTransformMixin'
]


class ImageBaseOperationsMixin(ABC):
    """Mixin with base operations on images
    """

    @staticmethod
    def _read_img(img_path: 'Path') -> 'ndarray':
        img = cv2.imread(str(img_path.resolve()))  # shape: (H, W, 3(BGR))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # shape: (H, W, 3(RGB))
        img = img.astype(np.float32)

        return img

    @staticmethod
    def _reorder_img_axes(img: 'ndarray',
                          axes_order: 'Tuple[int, ...]') -> 'ndarray':
        """Reorders axes in given `img` array

        Example:
            Reorder image in HWC format to CHW

            >>> img = np.random.randint(low=0, high=255, size=(64, 64, 3))
            >>> img_mod = self._reorder_img_axes(img, axes_order=(2, 0, 1))
            >>> img_mod.shape
            (3, 12, 12)

        Args:
            img: input image
            axes_order: new order of axes

        Returns:
            Image array with new order of axes
        """
        img_mod = np.transpose(img, axes_order)

        return img_mod

    @staticmethod
    def _extend_img_channels(img: 'ndarray',
                             channels_out: int = 3) -> 'ndarray':
        """Adds new shape at the end of `img` array and stack it `channels_out` times

        Args:
            img: input image with expected shape: (H, W)
            channels_out: number of channels to add

        Returns:
            Image array with new shape
        """
        img_mod = img.reshape(*img.shape, 1)  # shape: (H, W, 1)

        if channels_out > 1:
            img_mod = img_mod.repeat(channels_out, -1)  # shape: (H, W, channels_out)

        return img_mod


class ImageScalingMixin(ABC):
    """Mixin with scaling operations on images like standardize, min-max scale, etc.
    """

    _mean: 'ndarray'
    _std: 'ndarray'
    _min: 'ndarray'
    _max: 'ndarray'

    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):

        if hyper_params.get('mean') is not None and hyper_params.get('std') is not None:
            self.mean, self.std = hyper_params['mean'], hyper_params['std']
        if hyper_params.get('min') is not None and hyper_params.get('max') is not None:
            self.min, self.max = hyper_params['min'], hyper_params['max']

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value: list):
        """Sets mean attribute

        Args:
            value: list with 1 or 3 elements
        """
        self._mean = self.__convert_mean_std_input(value, 'mean')

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value: list):
        """Sets std attribute

        Args:
            value: list with 1 or 3 elements
        """
        self._std = self.__convert_mean_std_input(value, 'std')

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value: 'Union[int, float]'):
        self._min = self.__convert_min_max_input(value, 'min')

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value: 'Union[int, float]'):
        self._max = self.__convert_min_max_input(value, 'max')

    @staticmethod
    def __convert_mean_std_input(value: list,
                                 param_name: str) -> 'ndarray':
        if len(value) == 1:
            value_array = np.array(value, dtype=np.float32).squeeze()
        elif len(value) == 3:
            value_array = np.array(value, dtype=np.float32).reshape((1, 1, 3))
        else:
            raise ValueError(f'Param "{param_name}" must have 1 or 3 elements')

        return value_array

    @staticmethod
    def __convert_min_max_input(value: 'Union[int, float]',
                                param_name: str) -> 'ndarray':
        if isinstance(value, (int, float)):
            value_array = np.array(value, dtype=np.float32)
        else:
            raise ValueError(f'Param "{param_name}" must be int or float')

        return value_array

    def _standard_scale_img(self, img: 'ndarray',
                            per_image: bool = False) -> 'ndarray':
        """Standardizes image

        Args:
            img: image with next axis order: height, width, channels (channels must be last)
            per_image: whether to standardize img with it's mean and std
                or global mean/std given through `init`

        Returns:
            Standardized image
        """
        if per_image:
            img_mod = (img - img.mean()) / img.std()
        else:
            img_mod = (img - self.mean) / self.std

        return img_mod

    def _min_max_scale_img(self, img: 'ndarray',
                           per_image: bool = True) -> 'ndarray':
        """Min-max scales image

        Args:
            img: input image
            per_image: whether to scale img with it's min/max or global min/max given through `init`

        Returns:
            Min-max scaled image
        """
        if per_image:
            min_val, max_val = np.min(img), np.max(img)
        else:
            min_val, max_val = self.min, self.max

        if min_val == max_val:
            img_mod = img - min_val
        else:
            img_mod = (img - min_val) / (max_val - min_val)

        return img_mod


class ImageMixingMixin(ABC):
    """Mixin with mixing operations on images like mixup, mixin, etc.
    """

    _mixup_weights: 'ndarray'

    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):

        self.mixup_weights = hyper_params['mixup_weights']

    @property
    def mixup_weights(self):
        return self._mixup_weights

    @mixup_weights.setter
    def mixup_weights(self, value: 'List[float]'):
        value_array = np.array(value, dtype=np.float32)

        if value_array.sum() != 1:
            raise ValueError(f'Sum of "mixup_weights" must be equal to 1, not: {value_array.sum()}')

        self._mixup_weights = value_array

    def mixup(self, images: 'Tuple[ndarray, ...]',
              labels: 'Tuple[ndarray, ...]') -> 'Tuple[ndarray, ndarray]':
        mixed_img = (np.stack(images, axis=-1) * self.mixup_weights).sum(axis=-1)
        mixed_labels = (np.stack(labels, axis=-1) * self.mixup_weights).sum(axis=-1)

        return mixed_img, mixed_labels

    # ToDo: develop mixin method
    # def mixin(self):
    #     raise NotImplementedError


class ImageTransformMixin(ABC):
    """Mixin with albumentations images transforms
    """

    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 albu_transforms: 'Compose' = None,
                 **_ignored):

        self.albu_transforms = albu_transforms

    def _transform_img(self, img: 'ndarray') -> 'ndarray':
        img_mod = self.albu_transforms(image=img)['image']

        return img_mod
