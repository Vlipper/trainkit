from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from trainkit.core.trainer import Trainer


class LRfinder:
    def __init__(self, trainer: 'Trainer', logs_path: Path, min_lr: float, max_lr: float,
                 num_lrs: int = 200, beta: float = 0.8):
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.model_name = self.trainer.model_name
        # redefine loader for continuous working
        self.dataloader = ContIterDataLoader(trainer.train_loader)
        self.logs_path = logs_path
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_lrs = num_lrs
        self.beta = beta

        # range_test vars
        self.lrs = np.geomspace(self.min_lr, self.max_lr, num_lrs)  # linspace
        self.logs = {'lr': [], 'loss': [], 'avg_loss': []}
        self.min_optimal_lr, self.max_optimal_lr = None, None

    def range_test(self):
        avg_loss = None
        self.model.train()

        for iter_step in tqdm(range(self.num_lrs), desc='finding LR', total=self.num_lrs):
            # apply new lr
            cur_lr = self.lrs[iter_step]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            # train step
            batch = next(self.dataloader)
            batch_loss = self.trainer.optim_wrapper(self.model.batch_step)(iter_step, batch)

            # calculate smoothed loss
            if iter_step == 0:
                avg_loss = batch_loss
            else:
                avg_loss = self.beta * avg_loss + (1 - self.beta) * batch_loss

            # ToDo: stop range_test if the loss is exploding
            # smoothed_loss = avg_loss / (beta ** batch_num)
            # if batch_num > 1 and smoothed_loss > 4 * best_loss:
            #     return log_lrs, losses
            # Record the best loss
            # if smoothed_loss < best_loss or batch_num == 1:
            #     best_loss = smoothed_loss

            # store values
            self.logs['lr'].append(cur_lr)
            self.logs['avg_loss'].append(avg_loss)
            self.logs['loss'].append(batch_loss)

        self.trainer.rollback_states()

    def find_optimal_borders(self, left_seq_len_threshold=4, right_seq_len_threshold=2,
                             gain_threshold=0.999) -> Tuple[float, float]:
        if len(self.logs['lr']) == 0:
            raise Exception('You must run range_test before')

        left_border_idx, right_border_idx = None, None
        avg_loss = np.array(self.logs['avg_loss'])
        first_gain = np.append(1, avg_loss[1:] / avg_loss[:-1])

        # find left border
        potential_left_bord_idx = \
            [i for i, value in enumerate(first_gain) if value < gain_threshold]
        seq_len = 0
        for idx, bord in enumerate(potential_left_bord_idx):
            if idx == 0:
                continue
            # check loss decreases in sequence
            # if seq_len > left_seq_len_threshold than loss starts decreasing
            if (bord - potential_left_bord_idx[idx - 1]) == 1:
                seq_len += 1
            else:
                seq_len = 0
            if seq_len >= left_seq_len_threshold:
                left_border_idx = bord - left_seq_len_threshold
                break

        # find right border
        potential_right_bord_idx_shifted = \
            [i for i, value in enumerate(first_gain[left_border_idx:]) if value > 1]
        seq_len = 0
        for idx, bord in enumerate(potential_right_bord_idx_shifted):
            if idx == 0:
                continue
            # check loss increases in sequence
            # if seq_len > right_seq_len_threshold than loss starts increasing
            if (bord - potential_right_bord_idx_shifted[idx - 1]) == 1:
                seq_len += 1
            else:
                seq_len = 0
            if seq_len >= right_seq_len_threshold:
                right_border_idx = bord + left_border_idx - right_seq_len_threshold
                break
        # right_border_idx = potential_right_bord_idx_shifted[0] + left_border_idx - 3

        self.min_optimal_lr = self.logs['lr'][left_border_idx]
        self.max_optimal_lr = self.logs['lr'][right_border_idx]

        return self.min_optimal_lr, self.max_optimal_lr

    def plot(self, out_mode='save', xlim_left=1e-10, xlim_right=5e-1):
        if len(self.logs['lr']) == 0:
            raise Exception('You must run range_test before')

        # restrict x-axis values
        loss, avg_loss, lr = [np.array(self.logs[i]) for i in ('loss', 'avg_loss', 'lr')]
        boarded_lr_mask = (lr > xlim_left) & (lr < xlim_right)
        loss_plot, avg_loss_plot, lr_plot = [i[boarded_lr_mask] for i in (loss, avg_loss, lr)]

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.semilogx(lr_plot, avg_loss_plot, '-', label='avg loss')
        plt.semilogx(lr_plot, loss_plot, '--', label='raw loss')
        plt.grid(True, color='0.85')
        ax.set(title=self.model_name, ylabel='loss', xlabel='lr')

        # plot optimal min/max dots
        min_max_optimal_lr = (self.min_optimal_lr, self.max_optimal_lr)
        if None not in min_max_optimal_lr:
            border_idx = np.argwhere(np.isin(lr_plot, min_max_optimal_lr)).squeeze()
            plt.semilogx(lr_plot[border_idx], avg_loss_plot[border_idx], 'o')
            plt.vlines(lr_plot[border_idx], *ax.get_ylim())

        if out_mode == 'show':
            plt.show()
        elif out_mode == 'save':
            plt.savefig(Path(self.logs_path, '{}.png'.format(self.model_name)))
        else:
            raise Exception('out_mode has not known value')


class ContIterDataLoader(object):
    """A wrapper for iterating `torch.utils.data.DataLoader` with the ability to reset
    itself while `StopIteration` is raised."""

    def __init__(self, data_loader, auto_reset=True):
        self.data_loader = data_loader
        self.auto_reset = auto_reset
        self._iterator = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self._iterator)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            data = next(self._iterator)

        return data
