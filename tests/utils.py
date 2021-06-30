from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.utils.data import Sampler

from trainkit.datasets import BaseDataset
from trainkit.core.models import BaseNet

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor
    from typing import Any, Dict, Literal, Optional, Tuple


class TwoMoonsDataset(BaseDataset):
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):
        super().__init__(run_params=run_params,
                         hyper_params=hyper_params)

        self.data, self.labels = self._mine_data(run_params['n_samples'],
                                                 run_params['noise_level'],
                                                 run_params['rand_seed'])

    def _mine_data(self, n_samples: int,
                   noise_level: float,
                   rand_seed: int) -> 'Tuple[ndarray, ndarray]':
        data, labels = make_moons(n_samples=n_samples, noise=noise_level, random_state=rand_seed)
        data = data.astype(np.float32)

        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        obj_feats, obj_label = self.data[idx], self.labels[idx]

        return obj_feats, obj_label


class TwoMoonsModel(BaseNet):
    def __init__(self, device: 'torch.device',
                 **_ignored):
        super().__init__(device, **_ignored)

        self.model = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(2, 16, bias=True)),
            ('nonlin1', nn.ReLU()),
            ('lin2', nn.Linear(16, 32, bias=False)),
            ('nonlin2', nn.ReLU()),
            ('clf', nn.Linear(32, 2, bias=False))
        ]))

        self.loss_fn = nn.CrossEntropyLoss()
        # self.metrics_fn =

    def forward(self, feats: 'Tensor') -> 'Tensor':
        logits = self.model(feats)

        return logits

    def batch_step(self, batch_idx: int,
                   batch: 'Any') -> 'Dict[str, Tensor]':
        feats, targets = batch  # feats is a Tensor with shape: [bs, 2]
        feats, targets = feats.to(self.device), targets.to(self.device)

        logits = self.forward(feats)
        batch_loss = self.loss_fn(logits, targets)

        # calc loss
        if batch_idx == 0:
            self.batch_obj_losses, self.batch_obj_metrics = [], []
        self.batch_obj_losses.append(batch_loss.detach().cpu().item())

        # calc metrics
        _, preds = logits.detach().max(dim=1)
        correct = (preds == targets).sum().item()
        accuracy = correct / len(targets)
        self.batch_obj_metrics.append(accuracy)

        return {'loss_backward': batch_loss}


# ToDo: put this class into trainkit
class SplitSampler(Sampler):
    def __init__(self, dataset: 'BaseDataset',
                 split_name: 'Literal["train", "val"]',
                 run_params: dict,
                 hyper_params: dict):
        super().__init__(None)

        self.split_name = split_name

        full_data_size = len(dataset)
        self.train_size = int(full_data_size * (1 - hyper_params['val_share']))
        self.shuffle_train = (split_name == 'train') and hyper_params['epoch_shuffle_train']
        self.data_idxs = self._split_data(full_data_size,
                                          run_params['rand_seed'],
                                          run_params['shuffle'])

    def _split_data(self, full_data_size: int,
                    seed: int,
                    shuffle: bool = True) -> np.ndarray:
        data_idxs = np.arange(full_data_size)
        if shuffle:
            data_idxs = self._shuffle(data_idxs, seed)

        if self.split_name == 'train':
            data_idxs = data_idxs[:self.train_size]
        elif self.split_name == 'val':
            data_idxs = data_idxs[self.train_size:]
        else:
            raise ValueError('Parameter split_name must be one of: "train", "val". '
                             f'Given value: {self.split_name}')

        return data_idxs

    @staticmethod
    def _shuffle(idxs: np.ndarray,
                 seed: 'Optional[int]' = None) -> np.ndarray:
        idxs = idxs.copy()

        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)

        return idxs

    def __len__(self):
        return len(self.data_idxs)

    def __iter__(self):
        if self.shuffle_train:
            self.data_idxs = self._shuffle(self.data_idxs)

        yield from self.data_idxs.tolist()
