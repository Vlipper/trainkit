from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
from torch.nn import Module

if TYPE_CHECKING:
    from trainkit.core.trainer import Trainer
    import torch


class BaseOperationsMixin(ABC):
    @abstractmethod
    def train_preps(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_step(self, batch_idx: int,
                   batch: Any) -> Dict[str, 'torch.Tensor']:
        """
        Method receives batch (dataloaders' unmodified output on CPU) and its index.
        It must calc loss/metrics and save them into lists: batch_obj_losses, batch_obj_metrics.

        Example:
            Output may looks like below.
            It saves loss and metrics per object to aggregate them at the end of epoch later

            >>> losses, metrics = torch.zeros(10), torch.zeros(10)
            >>> batch_loss = losses.mean()

            >>> if batch_idx == 0:
            >>>     self.batch_obj_losses, self.batch_obj_metrics = [], []
            >>> self.batch_obj_losses.extend(losses)
            >>> self.batch_obj_metrics.extend(metrics)
            >>> out = {'loss_backward': batch_loss}
            >>> return out

        Args:
            batch_idx: index of given batch
            batch: batch of data (`torch.Tensor`), may be wrapped in dict, list

        Returns:
            Arbitrary dict, but it must contain key `loss_backward` with backwardable value of type
            `torch.Tensor`
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


class BaseNet(BaseOperationsMixin, Module, ABC):
    trainer: 'Trainer'
    batch_obj_losses: list
    batch_obj_metrics: list

    def __init__(self, device: 'torch.device',
                 **_ignored):
        super().__init__()

        self.device = device

    def train_preps(self, trainer: 'Trainer'):
        self.trainer = trainer
        self.to(self.device)
        self.train()

    def on_train_batch_end(self):
        # change lr if scheduler is cyclic
        if (self.trainer.lr_scheduler is not None) and self.trainer.lr_scheduler.is_cyclic:
            self.trainer.log_writer.write_scalar('lr/train',
                                                 self.trainer.optimizer.param_groups[0]['lr'],
                                                 self.trainer.n_iter_train)
            self.trainer.lr_scheduler.step()

    def on_train_part_end(self):
        # write mean values over batches into logs
        self.trainer.log_writer.write_scalar('losses/train',
                                             np.mean(self.batch_obj_losses).item(),
                                             self.trainer.epoch)
        self.trainer.log_writer.write_scalar('metrics/train',
                                             np.mean(self.batch_obj_metrics).item(),
                                             self.trainer.epoch)

    def on_val_part_end(self):
        # write mean values over batches into logs
        self.trainer.val_loss = np.mean(self.batch_obj_losses).item()
        self.trainer.val_metrics = np.mean(self.batch_obj_metrics).item()

        self.trainer.log_writer.write_scalar('losses/val',
                                             self.trainer.val_loss,
                                             self.trainer.epoch)
        self.trainer.log_writer.write_scalar('metrics/val',
                                             self.trainer.val_metrics,
                                             self.trainer.epoch)

        # change lr if scheduler is not cyclic
        if (self.trainer.lr_scheduler is not None) and (not self.trainer.lr_scheduler.is_cyclic):
            self.trainer.log_writer.write_scalar('lr/val',
                                                 self.trainer.optimizer.param_groups[0]['lr'],
                                                 self.trainer.epoch)
            self.trainer.lr_scheduler.step(self.trainer.val_loss)
