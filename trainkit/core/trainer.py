import logging
import functools
import gc
import os
from copy import deepcopy
from math import inf
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from trainkit.core.logger import LogWriter
from trainkit.core.lr_finder import LRFinder
from trainkit.utils.trainer_utils import LRSchedulerFactory, OptimizerFactory

if TYPE_CHECKING:
    from typing import Callable, Optional
    from trainkit.core.models import BaseNet
    from torch.utils.data import DataLoader

_logger = logging.getLogger(__name__)


class Trainer:
    log_writer: 'LogWriter'

    def __init__(self, model: 'BaseNet',
                 train_loader: 'DataLoader',
                 val_loader: 'DataLoader',
                 run_params: dict,
                 hyper_params: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_params = run_params
        self.hyper_params = hyper_params

        # paths params
        self.models_dir_path = run_params['paths'].get('models_path')

        # general params
        self.device = run_params['general']['device']
        self.model_name = run_params['general']['model_name']
        self.num_epochs = hyper_params['general']['num_epochs']
        self.clip_grad_norm_max = hyper_params['general'].get('clip_grad_norm_max')
        self.clip_grad_norm_type = hyper_params['general'].get('clip_grad_norm_type')
        self.metrics_comp_mode = run_params['general']['metrics_comparison_mode']
        self.save_last_state = run_params['general']['save_last_state']
        self.save_best_state = run_params['general']['save_best_state']
        self.log_writer = LogWriter(tboard_dir_path=run_params['paths']['tboard_path'],
                                    hparam_dir_path=run_params['paths']['hparam_path'],
                                    model_name=self.model_name)

        # find lr params
        self.is_find_lr = run_params['find_lr']['is_flr']
        self.find_lr_params = run_params['find_lr']

        # define optimizer and cache
        self.optimizer = OptimizerFactory.get_optimizer(
            model,
            hyper_params['optimizer']['name'],
            hyper_params['optimizer']['kwargs'])
        self.cache = self.cache_states()

        # skeleton vars
        self.epoch, self.n_iter_train = 0, 0
        self.batches_per_epoch = None
        self.lr_scheduler = None
        self.val_loss, self.val_metrics = None, None
        self.best_val_loss = inf
        self.best_val_metrics = inf if self.metrics_comp_mode == 'min' else -inf
        self.best_val_loss_epoch, self.best_val_metrics_epoch = None, None

    def run_find_lr(self):
        """Runs LR range test (find LR) and finds optimal lr.

        If `apply_optim_borders_flag` value in config is `True`, then optimal LR will be applied.
        """
        lr_finder = LRFinder(trainer=self, **self.find_lr_params['kwargs'])
        min_lr, max_lr, optimal_lr = lr_finder(**self.find_lr_params['kwargs'])

        if self.find_lr_params['is_flr_only']:
            os._exit(0)

        if self.find_lr_params['apply_optim_borders_flag']:
            self._apply_new_lr(optimal_lr)
            self.hyper_params['lr_scheduler']['kwargs'].update({'min_lr': min_lr, 'max_lr': max_lr})
            _logger.info('New min|max LRs were applied, their values: %.2e | %.2e', min_lr, max_lr)

        del lr_finder
        gc.collect()

    def _apply_new_lr(self, new_lr: float):
        """Updates learning rate in hyper params, initializes new optimizer and cache its state.

        Args:
            new_lr: learning rate to apply
        """
        self.hyper_params['optimizer']['kwargs'].update({'lr': new_lr})
        self.optimizer = OptimizerFactory.get_optimizer(
            self.model,
            self.hyper_params['optimizer']['name'],
            self.hyper_params['optimizer']['kwargs'])
        self.cache.update({'optimizer_state': deepcopy(self.optimizer.state_dict())})

        _logger.info('New LR was applied, its value: %.2e', new_lr)

    def pretrain_routine(self):
        self.model.train_preps(self)

        if self.is_find_lr:
            self.run_find_lr()

        self.batches_per_epoch = len(self.train_loader)  # it is needed for cyclic schedulers
        self.lr_scheduler = LRSchedulerFactory.get_scheduler(
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            batches_per_epoch=self.batches_per_epoch,
            lr_scheduler_name=self.hyper_params['lr_scheduler']['name'],
            lr_scheduler_kwargs=self.hyper_params['lr_scheduler']['kwargs'])

        self.log_writer.write_hparams(self.hyper_params)

    def fit(self):
        self.pretrain_routine()

        epoch_tqdm = tqdm(total=self.num_epochs, desc='epochs', leave=True)
        for self.epoch in range(self.num_epochs):
            # train part
            self.model.train()
            for batch_idx, batch in tqdm(enumerate(self.train_loader),
                                         total=len(self.train_loader),
                                         desc='train', leave=False):
                _ = self.optim_wrapper(self.model.batch_step)(batch_idx, batch)
                self.model.on_train_batch_end()
                self.n_iter_train += 1
            self.model.on_train_part_end()

            # val part
            # ToDo: ? if self.val_loader is not None: ?
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(self.val_loader),
                                             total=len(self.val_loader),
                                             desc='val', leave=False):
                    self.model.batch_step(batch_idx, batch)
            self.model.on_val_part_end()

            # update best loss and metrics
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.best_val_loss_epoch = self.epoch

            new_metrics_better = self._is_new_metrics_better()
            if new_metrics_better:
                self.best_val_metrics = self.val_metrics
                self.best_val_metrics_epoch = self.epoch

            # save states
            if self.save_last_state or self.save_best_state:
                if self.lr_scheduler is not None:
                    lr_scheduler_state = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state = None
                save_kw = {'val_loss': self.val_loss,
                           'val_metrics': self.val_metrics,
                           'lr_scheduler_state': lr_scheduler_state}

                # save last state
                if self.save_last_state:
                    save_path = Path(self.models_dir_path, f'{self.model_name}_last.pth')
                    self.save_states(save_path, **save_kw)

                # save best state
                if self.save_best_state and new_metrics_better:
                    save_path = Path(self.models_dir_path, f'{self.model_name}_best_metrics.pth')
                    self.save_states(save_path, **save_kw)

            epoch_tqdm.update()
        epoch_tqdm.close()

        hparams = {'hparams': self.hyper_params.copy()}
        results = {'best_loss': self.best_val_loss,
                   'best_metrics': self.best_val_metrics,
                   'best_loss_epoch': self.best_val_loss_epoch,
                   'best_metrics_epoch': self.best_val_metrics_epoch}
        hparams.update({'results': results})
        self.log_writer.write_hparams(hparams=hparams)
        self.log_writer.tb_writer.close()

    def optim_wrapper(self, train_step: 'Callable') -> 'Callable[..., float]':
        """
        Decorator for model step with zero_grad, backward, optimizer step

        Args:
            train_step: model step over batch:
                forward, calc loss, must return dict with 'backward_loss' to make optim step
        Returns:
            Wrapped train_step
        """
        @functools.wraps(train_step)
        def wrapper(*args, **kwargs) -> float:
            self.optimizer.zero_grad()
            out = train_step(*args, **kwargs)
            batch_loss = out['loss_backward']
            batch_loss.backward()
            if self.clip_grad_norm_max and self.clip_grad_norm_type:
                _ = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                   max_norm=self.clip_grad_norm_max,
                                                   norm_type=self.clip_grad_norm_type)
            self.optimizer.step()

            return batch_loss.item()

        return wrapper

    def _is_new_metrics_better(self) -> bool:
        if self.metrics_comp_mode == 'min':
            new_metrics_better = self.val_metrics < self.best_val_metrics
        elif self.metrics_comp_mode == 'max':
            new_metrics_better = self.val_metrics > self.best_val_metrics
        else:
            raise ValueError("Param 'metrics_comparison_mode' must be 'min' or 'max'. "
                             f"Given value: '{self.metrics_comp_mode}'")

        return new_metrics_better

    def save_states(self, save_path: Path, **kwargs):
        state_dict = {'epoch': self.epoch,
                      'n_iter_train': self.n_iter_train,
                      'model_state': self.model.state_dict(),
                      'optimizer_state': self.optimizer.state_dict(),
                      'run_params': self.run_params,
                      'hyper_params': self.hyper_params}
        state_dict.update(kwargs)

        torch.save(state_dict, save_path)

    def cache_states(self):
        cache_dict = {'model_state': deepcopy(self.model.state_dict()),
                      'optimizer_state': deepcopy(self.optimizer.state_dict())}
        # if self.lr_scheduler is not None:
        #     cache_dict['lr_scheduler_state'] = deepcopy(self.lr_scheduler.state_dict())

        return cache_dict

    def rollback_states(self):
        self.model.load_state_dict(self.cache['model_state'])
        self.optimizer.load_state_dict(self.cache['optimizer_state'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self.cache['lr_scheduler_state'])
