from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

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
    def batch_step(self, *args, **kwargs) -> Dict[str, 'torch.Tensor']:
        """
        Method must receive data batch (cpu), forward it, calc loss/metrics and save them into
        lists: batch_obj_losses, batch_obj_metrics.

        Returns:
            Arbitrary dict, but it must contain key `loss_backward` with `torch.Tensor` value.
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


class BaseNet(BaseOperationsMixin, Module):
    trainer: 'Trainer'
    batch_obj_losses: list
    batch_obj_metrics: list

    def __init__(self, device: 'torch.device',
                 **_ignored):
        super().__init__()

        self.device = device

    @abstractmethod
    def batch_step(self, *args, **kwargs) -> Dict[str, 'torch.Tensor']:
        pass

    def train_preps(self, trainer: 'Trainer'):
        self.trainer = trainer
        self.to(self.device)
        self.train()

    def on_train_batch_end(self):
        # change lr if scheduler is cyclic
        if (self.trainer.lr_scheduler is not None) and self.trainer.lr_scheduler.is_cyclic:
            self.trainer.log_writer.write_log('lr/train',
                                              self.trainer.optimizer.param_groups[0]['lr'],
                                              self.trainer.n_iter_train)
            self.trainer.lr_scheduler.step()

    def on_train_part_end(self):
        # write mean values over batches into logs
        self.trainer.log_writer.write_log('losses/train',
                                          np.mean(self.batch_obj_losses),
                                          self.trainer.epoch)
        self.trainer.log_writer.write_log('metrics/train',
                                          np.mean(self.batch_obj_metrics),
                                          self.trainer.epoch)

    def on_val_part_end(self):
        # write mean values over batches into logs
        self.trainer.val_loss = np.mean(self.batch_obj_losses)
        self.trainer.val_metrics = np.mean(self.batch_obj_metrics)

        self.trainer.log_writer.write_log('losses/val',
                                          self.trainer.val_loss,
                                          self.trainer.epoch)
        self.trainer.log_writer.write_log('metrics/val',
                                          self.trainer.val_metrics,
                                          self.trainer.epoch)

        # change lr if scheduler is not cyclic
        if (self.trainer.lr_scheduler is not None) and (not self.trainer.lr_scheduler.is_cyclic):
            self.trainer.log_writer.write_log('lr/val',
                                              self.trainer.optimizer.param_groups[0]['lr'],
                                              self.trainer.epoch)
            self.trainer.lr_scheduler.step(self.trainer.val_loss)
