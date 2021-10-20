import gc
import sys
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


class Trainer:
    log_writer: 'Optional[LogWriter]'
    train_loader: 'DataLoader'
    val_loader: 'DataLoader'

    def __init__(self, model: 'BaseNet',
                 run_params: dict,
                 hyper_params: dict):
        self.model = model
        self.run_params = run_params
        self.hyper_params = hyper_params

        # paths params
        self.models_dir_path = run_params['paths'].get('models_path')
        self.tboard_dir_path = run_params['paths'].get('tboard_path')
        self.hparam_dir_path = run_params['paths'].get('hparam_path')

        # general params
        self.device = run_params['general']['device']
        self.model_name = run_params['general']['model_name']
        self.num_epochs = hyper_params['general']['num_epochs']
        self.clip_grad_norm_max = hyper_params['general'].get('clip_grad_norm_max')
        self.clip_grad_norm_type = hyper_params['general'].get('clip_grad_norm_type')
        self.metrics_comp_mode = run_params['general']['metrics_comparison_mode']
        self.save_last_state = run_params['general']['save_last_state']
        self.save_best_state = run_params['general']['save_best_state']
        self.save_logs = run_params['general']['save_logs']

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
        self.log_writer = None

    def run_find_lr(self):
        """Runs LR range test (find LR) and finds optimal lr.

        If `apply_optim_borders_flag` value in config is `True`, then optimal LR will be applied.
        """
        lr_finder = LRFinder(trainer=self, **self.find_lr_params['kwargs'])
        min_lr, max_lr, optimal_lr = lr_finder(**self.find_lr_params['kwargs'])

        if self.find_lr_params['is_flr_only']:
            sys.exit()

        if self.find_lr_params['apply_optim_borders_flag']:
            self._apply_new_lr(optimal_lr)
            self.hyper_params['lr_scheduler']['kwargs'].update({'min_lr': min_lr, 'max_lr': max_lr})
        del lr_finder
        gc.collect()

    @staticmethod
    def _calc_optimal_lr(min_lr: float,
                         max_lr: float) -> float:
        """Calculates optimal learning rate between given min and max borders

        Args:
            min_lr: minimal learning rate border
            max_lr: maximum learning rate border

        Returns:
            learning rate between min and max borders
        """
        optimal_lr = min_lr + 0.5 * (max_lr - min_lr)  # init new lr between min_lr and max_lr

        return optimal_lr

    def _apply_new_lr(self, new_lr: float):
        """
        Update learning rate value in hyper params, init new optimizer and cache new optimizer state

        Args:
            new_lr: learning rate to apply
        """
        self.hyper_params['optimizer']['kwargs'].update({'lr': new_lr})

        self.optimizer = OptimizerFactory.get_optimizer(
            self.model,
            self.hyper_params['optimizer']['name'],
            self.hyper_params['optimizer']['kwargs'])

        self.cache.update({'optimizer_state': deepcopy(self.optimizer.state_dict())})

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

        if self.save_logs:
            self.log_writer = LogWriter(tboard_dir_path=self.tboard_dir_path,
                                        hparam_dir_path=self.hparam_dir_path,
                                        model_name=self.model_name)
            self.log_writer.write_hparams(self.hyper_params)

    def fit(self, train_loader: 'DataLoader',
            val_loader: 'DataLoader'):
        self.train_loader, self.val_loader = train_loader, val_loader
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

        if self.log_writer is not None:
            hparams = {'hparams': self.hyper_params.copy()}
            results = {'best_loss': self.best_val_loss,
                       'best_metrics': self.best_val_metrics,
                       'best_loss_epoch': self.best_val_loss_epoch,
                       'best_metrics_epoch': self.best_val_metrics_epoch}
            hparams.update({'results': results})

            self.log_writer.write_hparams(hparams=hparams)
            self.log_writer.close()

    def optim_wrapper(self, train_step: 'Callable') -> 'Callable':
        """
        Decorator for model step with zero_grad, backward, optimizer step

        Args:
            train_step: model step over batch:
                forward, calc loss, must return dict with 'backward_loss' to make optim step
        Returns:
            Wrapped train_step
        """
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

    # def update_params(self, new_params: dict, params_group: Optional[str] = None):
    #     for key, value in new_params.items():
    #         # update existing params
    #         if key in self.run_params:
    #             self.run_params.update({key: value})
    #         elif key in self.hyper_params:
    #             self.hyper_params.update({key: value})
    #         # add new params (arg params_group required)
    #         else:
    #             if params_group == 'run_params':
    #                 self.run_params.update({key: value})
    #             elif params_group == 'hyper_params':
    #                 self.hyper_params.update({key: value})
    #             else:
    #                 raise Exception('Keys in given new_params are not in one of existing group or '
    #                                 'params_group has not known value: "run_params", "hyper_params"')
    #         self.all_params.update({key: value})


class CrossVal:
    def __init__(self):
        raise NotImplementedError
#
#     def train(self, train_val_sets_pairs: list, agg_logs=True):
#         cv_size = len(train_val_sets_pairs)
#         cv_tqdm = tqdm(total=cv_size, desc='cross-val folds')
#         model_name_raw = self.model_name
#         for fold in range(cv_size):
#             self.n_iter_train, self.best_val_metrics = 0, 0
#             self.model_name = '{}_fold_{}'.format(model_name_raw, fold)
#             # load initial states
#             self.model.load_state_dict(self.cache['model_state'], strict=False)
#             self.optimizer.load_state_dict(self.cache['optimizer_state'])
#             # train model on fold
#             train_dataset, val_dataset = train_val_sets_pairs[fold]
#             self.train(train_dataset, val_dataset)
#             self.lr_scheduler = None
#             cv_tqdm.update()
#         cv_tqdm.close()
#         self.model_name = model_name_raw
#
#         if agg_logs:
#             self.write_cv_agg_logs(cv_size)
#
#     def cv_blend(self, cv_size: int, blend_dataset, agg_logs=True):
#         # inference cv models
#         cv_preds, blend_targets = None, None
#         blend_loader = self.get_dataloader(blend_dataset, is_val=True)
#         for fold in tqdm(range(cv_size), desc='cv models blending'):
#             # load model state
#             model_state_filename = '{}_fold_{}_best_metrics_state.pth'.format(self.model_name, fold)
#             model_state_path = Path(self.models_dir_path, model_state_filename)
#             model_state = torch.load(model_state_path, torch.device('cpu'))['model_state']
#             _ = self.model.load_state_dict(model_state, strict=False)
#
#             # get scores
#             blend_preds, blend_targets = torch.tensor([]), torch.tensor([])
#             for batch in blend_loader:
#                 blend_preds, blend_targets = self.val_batch(batch, blend_preds, blend_targets,
#                                                             use_sigmoid=True)
#             # stack scores
#             blend_preds = blend_preds[:, 0]
#             if fold == 0:
#                 cv_preds = blend_preds.view(-1, 1)
#             else:
#                 cv_preds = torch.cat([cv_preds, blend_preds.view(-1, 1)], dim=1)
#
#         # blend scores
#         blended_preds = cv_preds.mean(dim=1)
#         blend_metrics = self.model.calc_metrics(blended_preds, blend_targets[:, 0])
#
#         if agg_logs:
#             self.write_cv_agg_logs(cv_size, metrics_dict={'hparam/metrics_blend': blend_metrics})
#
#     def write_cv_agg_logs(self, cv_size: int, metrics_dict=None):
#         scalars_tags = None
#         fold_tag_val_dict = {}
#         for fold in range(cv_size):
#             fold_name = 'fold_{}'.format(fold)
#             fold_tag_val_dict.update({fold_name: {}})
#
#             # define and read fold's logs
#             fold_logs_path = str(self.cv_log_paths[fold])
#             # ToDo: second arg can be modified for less data reading
#             acc = event_accumulator.EventAccumulator(fold_logs_path, {event_accumulator.SCALARS: 0})
#             acc = acc.Reload()
#             scalars_tags = acc.scalars.Keys()
#             for tag in scalars_tags:
#                 step_val = [(event.step, event.value) for event in acc.Scalars(tag)]
#                 fold_tag_val_dict[fold_name].update({tag: step_val})
#
#         # write aggregated logs
#         self.init_log_writer(postfix='_folds_mean')
#         for tag in scalars_tags:
#             # ToDo: костылина из-за различающихся размеров фолдов
#             num_events = min([len(fold_tag_val_dict['fold_{}'.format(fold)][tag])
#                               for fold in range(cv_size)])
#
#             for event_idx in range(num_events):
#                 folds_vals = [fold_tag_val_dict['fold_{}'.format(fold)][tag][event_idx][1]
#                               for fold in range(cv_size)]
#                 step = fold_tag_val_dict['fold_0'][tag][event_idx][0]
#                 self.write_scalars(tag, np.mean(folds_vals), step)
#
#         # write hyperparams logs
#         if metrics_dict is None:
#             metrics_dict = {}
#
#         folds_max_val_metrics, folds_min_val_losses = [], []
#         for fold in range(cv_size):
#             max_val_metrics = max([val for step, val in
#                                    fold_tag_val_dict['fold_{}'.format(fold)]['metrics/val']])
#             folds_max_val_metrics.append(max_val_metrics)
#             min_val_loss = min([val for step, val in
#                                 fold_tag_val_dict['fold_{}'.format(fold)]['losses/val']])
#             folds_min_val_losses.append(min_val_loss)
#
#         metrics_dict.update({'hparam/metrics_best_folds': np.mean(folds_max_val_metrics),
#                              'hparam/loss_best_folds': np.mean(folds_min_val_losses)})
#         hparam_dict = dict([(key, val) for key, val in self.h_params.items() if val is not None])
#         self.log_writer.add_hparams(hparam_dict, metrics_dict)
