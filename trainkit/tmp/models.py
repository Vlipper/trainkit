from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from comp_utils.utils.training_utils import Trainer


class NetBaseMixin(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.device = kwargs['device']
        self.loss_fn = kwargs.get('loss_fn')
        self.metrics_fn = kwargs.get('metrics_fn')

        self.trainer: Optional['Trainer'] = None
        self.batch_obj_losses: Optional[list] = None
        self.batch_obj_metrics: Optional[list] = None

    def train_preps(self, trainer):
        self.trainer = trainer
        self.to(self.device)
        self.train()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def batch_step(self, *args, **kwargs):
        raise NotImplementedError

    def on_train_batch_end(self):
        # change lr if scheduler is cyclic  ToDo: ? перевести на эпохи ?
        if (self.trainer.lr_scheduler is not None) and self.trainer.lr_scheduler.is_cyclic:
            self.trainer.log_writer.write_log(
                'lr/train', self.trainer.optimizer.param_groups[0]['lr'], self.trainer.n_iter_train)
            self.trainer.lr_scheduler.step(None)

    def on_train_part_end(self):
        # write mean values over batches into logs
        self.trainer.log_writer.write_log(
            'losses/train', np.mean(self.batch_obj_losses), self.trainer.epoch)
        self.trainer.log_writer.write_log(
            'metrics/train', np.mean(self.batch_obj_metrics), self.trainer.epoch)

    def on_val_part_end(self):
        # write mean values over batches into logs
        self.trainer.val_loss = np.mean(self.batch_obj_losses)
        self.trainer.val_metrics = np.mean(self.batch_obj_metrics)

        self.trainer.log_writer.write_log('losses/val', self.trainer.val_loss, self.trainer.epoch)
        self.trainer.log_writer.write_log(
            'metrics/val', self.trainer.val_metrics, self.trainer.epoch)

        # change lr if scheduler is not cyclic
        if (self.trainer.lr_scheduler is not None) and (not self.trainer.lr_scheduler.is_cyclic):
            self.trainer.log_writer.write_log(
                'lr/val', self.trainer.optimizer.param_groups[0]['lr'], self.trainer.epoch)
            self.trainer.lr_scheduler.step(self.trainer.val_loss)

    def _calc_loss(self, logits, targets) -> torch.Tensor:
        if self.loss_fn is None:
            raise Exception('loss_fn cannot be None')

        loss = self.loss_fn(logits, targets)

        return loss

    def _calc_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        if self.metrics_fn is None:
            raise Exception('loss_fn cannot be None')

        preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
        metrics = self.metrics_fn(preds, targets)

        return metrics


# class SpecNet(NetBaseMixin):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
