from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import _LRScheduler


class CstmOneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_lr,  # anneal_strategy,
                 step_frac_up, step_frac_down, epochs, steps_per_epoch,
                 init_lr_div_factor, min_lr_div_factor, last_epoch=-1):
        # sanity checks
        if not (1 >= (step_frac_up + step_frac_down) >= 0):
            raise ValueError('step_frac_up + step_frac_down must be between 0 and 1')

        self.optimizer = optimizer
        self.max_lr = max_lr
        # self.anneal_strategy = anneal_strategy # 'linear', 'cos'
        self.step_frac_up = step_frac_up
        self.step_frac_down = step_frac_down
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.init_lr_div_factor = init_lr_div_factor
        self.min_lr_div_factor = min_lr_div_factor

        self.init_lr = max_lr / self.init_lr_div_factor
        self.min_lr = self.init_lr / self.min_lr_div_factor
        self.total_steps = epochs * steps_per_epoch

        interp_x = [0, step_frac_up * self.total_steps,
                    (step_frac_up + step_frac_down) * self.total_steps,
                    self.total_steps]
        interp_y = [self.init_lr, max_lr, self.init_lr, self.min_lr]
        self.interp_func = interp1d(x=interp_x, y=interp_y, kind='linear')

        super(CstmOneCycleLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for _ in self.optimizer.param_groups:
            group_lr = self.interp_func(step_num)
            lrs.append(float(group_lr))

        return lrs
