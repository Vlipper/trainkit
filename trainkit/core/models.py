from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from torch.nn import Module

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional
    import torch
    from torch import Tensor
    from trainkit.core.trainer import Trainer


class BaseOperationsMixin(ABC):
    @abstractmethod
    def train_preps(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_step(self, batch_idx: int,
                   batch: 'Any') -> 'Dict[str, Tensor]':
        """
        Method receives batch (dataloaders' unmodified output on CPU) and its index.
        It must calc loss/metrics and save them into lists: batch_losses, batch_metrics.

        Example:
            Output may looks like below.
            It saves loss and metrics per object to aggregate them at the end of epoch later

            >>> losses, metrics = torch.zeros(10), torch.zeros(10)
            >>> batch_loss = losses.mean()

            >>> if batch_idx == 0:
            >>>     self.batch_losses, self.batch_metrics = [], []
            >>> self.batch_losses.extend(losses)
            >>> self.batch_metrics.extend(metrics)
            >>> out = {'loss_backward': batch_loss}

            >>> return out

        Args:
            batch_idx: index of given batch
            batch: batch of data (`torch.Tensor`), may be wrapped in dict, list

        Returns:
            Arbitrary dict, but it must contain key `loss_backward` with backwardable `Tensor` value
        """
        pass

    @abstractmethod
    def on_train_batch_end(self):
        pass

    @abstractmethod
    def on_train_part_end(self):
        pass

    @abstractmethod
    def on_val_part_end(self):
        pass


class BaseNet(BaseOperationsMixin,
              Module,
              ABC):
    trainer: 'Trainer'
    batch_losses: list
    batch_metrics: list

    def __init__(self, device: 'torch.device',
                 loss_agg_fn: 'Optional[Callable[[list], float]]' = None,
                 metrics_agg_fn: 'Optional[Callable[[list], float]]' = None,
                 **_ignored):
        """

        Args:
            device: torch.device to use model on it
            loss_agg_fn: function to aggregate batch_losses list, which may extend at `batch_step`
            metrics_agg_fn: function to aggregate batch_metrics list, which may extend at `batch_step`
        """
        super().__init__()

        self.device = device
        self.loss_agg_fn = lambda x: np.mean(x).item() if loss_agg_fn is None else loss_agg_fn
        self.metrics_agg_fn = lambda x: np.mean(x).item() if metrics_agg_fn is None else metrics_agg_fn

    def train_preps(self, trainer: 'Trainer'):
        self.trainer = trainer
        self.to(self.device)
        self.train()

    def on_train_batch_end(self):
        # change lr if scheduler is cyclic
        if self.trainer.lr_scheduler is not None and self.trainer.lr_scheduler.is_cyclic:
            if self.trainer.log_writer is not None:
                self.trainer.log_writer.write_scalar('lr/train',
                                                     self.trainer.optimizer.param_groups[0]['lr'],
                                                     self.trainer.n_iter_train)
            self.trainer.lr_scheduler.step()

    def on_train_part_end(self):
        # write mean values over batches into logs
        if self.trainer.log_writer is not None:
            self.trainer.log_writer.write_scalar('losses/train',
                                                 self.loss_agg_fn(self.batch_losses),
                                                 self.trainer.epoch)
            self.trainer.log_writer.write_scalar('metrics/train',
                                                 self.metrics_agg_fn(self.batch_metrics),
                                                 self.trainer.epoch)

    def on_val_part_end(self):
        # write mean values over batches into logs
        self.trainer.val_loss = self.loss_agg_fn(self.batch_losses)
        self.trainer.val_metrics = self.metrics_agg_fn(self.batch_metrics)

        if self.trainer.log_writer is not None:
            self.trainer.log_writer.write_scalar('losses/val',
                                                 self.trainer.val_loss,
                                                 self.trainer.epoch)
            self.trainer.log_writer.write_scalar('metrics/val',
                                                 self.trainer.val_metrics,
                                                 self.trainer.epoch)

        # change lr if scheduler is not cyclic
        if self.trainer.lr_scheduler is not None and (not self.trainer.lr_scheduler.is_cyclic):
            if self.trainer.log_writer is not None:
                self.trainer.log_writer.write_scalar('lr/val',
                                                     self.trainer.optimizer.param_groups[0]['lr'],
                                                     self.trainer.epoch)
            self.trainer.lr_scheduler.step(self.trainer.val_loss)
