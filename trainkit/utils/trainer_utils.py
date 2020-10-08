import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from trainkit.core.lr_scheduler import CstmOneCycleLR


class OptimizerFactory:
    @staticmethod
    def get_optimizer(model: nn.Module, optimizer_name: str, optimizer_kwargs: dict):
        if optimizer_name == 'sgd':
            optimizer = SGD(model.parameters(), **optimizer_kwargs)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise ValueError('Given optimizer_name: "{}" is not known'.format(optimizer_name))

        return optimizer


class LRSchedulerFactory:
    @classmethod
    def get_scheduler(cls, optimizer: Optimizer, num_epochs: int, batches_per_epoch: int,
                      lr_scheduler_name: str, lr_scheduler_kwargs: dict):
        kwargs_mod = lr_scheduler_kwargs.copy()
        kwargs_mod.update({'num_epochs': num_epochs, 'batches_per_epoch': batches_per_epoch})

        if lr_scheduler_name is None:
            lr_scheduler = None
        elif lr_scheduler_name == 'rop':
            lr_scheduler = cls.__get_rop_scheduler(optimizer, **kwargs_mod)
        elif lr_scheduler_name == 'clr':
            lr_scheduler = cls.__get_clr_scheduler(optimizer, **kwargs_mod)
        elif lr_scheduler_name == 'cst_onecycle':
            lr_scheduler = cls.__get_cst_onecycle_scheduler(optimizer, **kwargs_mod)
        else:
            raise ValueError('Given lr_scheduler_name: "{}" is not known'.format(lr_scheduler_name))

        return lr_scheduler

    @staticmethod
    def __get_rop_scheduler(optimizer, min_lr, max_lr, **kwargs):
        # init new lr between min_lr and max_lr
        rop_init_lr = min_lr + 0.5 * (max_lr - min_lr)
        for group in optimizer.param_groups:
            group['lr'] = rop_init_lr

        lr_scheduler = ReduceLROnPlateau(optimizer, mode=kwargs['mode'], factor=kwargs['factor'],
                                         threshold=kwargs['threshold'], patience=kwargs['patience'],
                                         threshold_mode=kwargs['threshold_mode'],
                                         cooldown=kwargs['cooldown'])
        setattr(lr_scheduler, 'is_cyclic', False)

        return lr_scheduler

    @staticmethod
    def __get_clr_scheduler(optimizer, min_lr, max_lr, **kwargs):
        clr_stepsize_up = int(kwargs['clr_step_size_up'] * kwargs['batches_per_epoch'])
        clr_stepsize_down = int(kwargs['clr_step_size_down'] * kwargs['batches_per_epoch'])
        clr_stepsize_up = clr_stepsize_up if clr_stepsize_up > 0 else 1
        clr_stepsize_down = clr_stepsize_down if clr_stepsize_down > 0 else 1

        lr_scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr,
                                step_size_up=clr_stepsize_up, step_size_down=clr_stepsize_down,
                                mode=kwargs['mode'], cycle_momentum=kwargs['cycle_momentum'])
        setattr(lr_scheduler, 'is_cyclic', True)

        return lr_scheduler

    @staticmethod
    def __get_cst_onecycle_scheduler(optimizer, max_lr, **kwargs):
        lr_scheduler = CstmOneCycleLR(optimizer, max_lr=max_lr, epochs=kwargs['num_epochs'],
                                      steps_per_epoch=kwargs['batches_per_epoch'],
                                      step_frac_up=kwargs['step_frac_up'],
                                      step_frac_down=kwargs['step_frac_down'],
                                      init_lr_div_factor=kwargs['init_lr_div_factor'],
                                      min_lr_div_factor=kwargs['min_lr_div_factor'])
        setattr(lr_scheduler, 'is_cyclic', True)

        return lr_scheduler
