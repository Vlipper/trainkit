from abc import ABC
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np

from trainkit.datasets.base import BaseDataset
from trainkit.datasets.properties import ImageDatasetProperties

if TYPE_CHECKING:
    from albumentations import Compose
    from pathlib import Path


class ImageBaseDataset(ImageDatasetProperties, BaseDataset, ABC):
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 albu_transforms: 'Compose' = None,
                 **_ignored):
        super().__init__(run_params=run_params,
                         hyper_params=hyper_params,
                         **_ignored)

        self.albu_transforms = albu_transforms
        if hyper_params.get('mean') is not None and hyper_params.get('std') is not None:
            self.mean, self.std = hyper_params['mean'], hyper_params['std']
        if hyper_params.get('min') is not None and hyper_params.get('max') is not None:
            self.min, self.max = hyper_params['min'], hyper_params['max']

    @staticmethod
    def _read_img(img_path: 'Path') -> np.ndarray:
        img = cv2.imread(str(img_path.resolve()))  # shape: (H, W, 3(BGR))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # shape: (H, W, 3(RGB))
        img = img.astype(np.float32)

        return img

    def _standard_scale_img(self, img: np.ndarray,
                            per_image: bool = False) -> np.ndarray:
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

    def _min_max_scale_img(self, img: np.ndarray,
                           per_image: bool = True) -> np.ndarray:
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

    @staticmethod
    def _reorder_img_axes(img: np.ndarray,
                          axes_order: Tuple[int, ...]) -> np.ndarray:
        """Reorders axes in given `img` array

        Example:
            Reorder image in HWC format to CHW

            >>> img = np.random.randint(low=0, high=255, size=(64, 64, 3))
            >>> img_mod = self._reorder_img_axes(img, axes_order=(2, 0, 1))
            >>> img_mod.shape
            (3, 12, 12)

        Args:
            img: image array to mod
            axes_order: new order of axes

        Returns:
            Image array with new order of axes
        """
        img_mod = np.transpose(img, axes_order)

        return img_mod

    @staticmethod
    def _extend_img_channels(img: np.ndarray) -> np.ndarray:
        """
        Adds new shape at the end of `img` array and repeat it three times on new shape

        Args:
            img: image array to mod

        Returns:
            Image array with new shape
        """
        img_mod = img.reshape(*img.shape, 1)  # shape: (H, W, 3)
        img_mod = img_mod.repeat(3, -1)  # shape: (H, W, 3)

        return img_mod

    def _transform_img(self, img: np.ndarray) -> np.ndarray:
        img_mod = self.albu_transforms(image=img)['image']

        return img_mod
