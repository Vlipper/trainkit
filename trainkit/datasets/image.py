from abc import ABC
from typing import TYPE_CHECKING, Tuple

import numpy as np
import cv2

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

    @staticmethod
    def _read_img(img_path: 'Path') -> np.ndarray:
        img = cv2.imread(img_path.as_posix())  # shape: (H, W, 3)
        img = img.astype(np.float32)

        return img

    def _standard_scale_img(self, img: np.ndarray) -> np.ndarray:
        """Standardizes image

        Args:
            img: image with next axis order: height, width, channels (channels must be last)

        Returns:
            Standardized image
        """
        img_mod = (img - self.mean) / self.std

        return img_mod

    @staticmethod
    def _min_max_scale_img(img: np.ndarray) -> np.ndarray:
        img_min, img_max = np.min(img), np.max(img)

        if img_min == img_max:
            img_mod = img - img_min
        else:
            img_mod = (img - img_min) / (img_max - img_min)

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
        if self.albu_transforms is None:
            raise ValueError('Image cannot be transformed without `albu_transforms` attribute')

        img_mod = self.albu_transforms(image=img)['image']

        return img_mod
