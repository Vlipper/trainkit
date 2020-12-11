from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader
    from trainkit.core.trainer import Trainer


class LRFinder:
    """
    LRFinder.

    Args:
        trainer: Trainer instance
        min_lr: minimum learning rate for range test
        max_lr: maximum learning rate for range test
        num_lrs: number of learning rates to check
        **_ignored: ignored kwargs storage
    """
    logs: Dict[str, Union[list, np.ndarray]]

    def __init__(self, trainer: 'Trainer',
                 min_lr: float = 1e-7,
                 max_lr: float = 0.9,
                 num_lrs: int = 200,
                 **_ignored):
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.model_name = self.trainer.model_name
        self.dataloader = ContinuousIterDataLoader(trainer.train_loader)
        self.num_lrs = num_lrs

        # vars for range_test
        self.lrs = np.geomspace(start=min_lr, stop=max_lr, num=num_lrs)
        self.logs = {'lr': [], 'loss': [], 'avg_loss': []}
        self.min_optimal_lr, self.max_optimal_lr = None, None
        self.best_avg_loss = None

    def range_test(self, smooth_beta: float = 0.8,
                   is_early_stopping: bool = True,
                   early_stopping_mult_factor: float = 3,
                   **_ignored):
        """
        Train model by iterating over continuous version of `trainer.train_loader`
        with applying new learning rate from `self.lrs` before every batch.
        After every batch step it calculates smoothed loss (`avg_loss`) and store into logs:
        lr, loss, avg_loss

        Args:
            smooth_beta: coefficient for smoothed loss calculation:
                `avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * batch_loss`
            is_early_stopping: check loss explosion and stop training
            early_stopping_mult_factor: multiplier value in which times current batch loss
                must be bigger than best smoothed loss to raise `StopIteration`
            **_ignored: ignored kwargs storage
        """
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
                avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * batch_loss

            if is_early_stopping:
                if self.best_avg_loss is None or self.best_avg_loss > avg_loss:
                    self.best_avg_loss = avg_loss

                try:
                    self.__early_stop_check(batch_loss, early_stopping_mult_factor)
                except StopIteration:
                    break

            # store values into logs
            self.logs['lr'].append(cur_lr)
            self.logs['avg_loss'].append(avg_loss)
            self.logs['loss'].append(batch_loss)

        self.logs.update({key: np.array(val) for key, val in self.logs.items()})
        self.trainer.rollback_states()

    def __early_stop_check(self, current_loss: float,
                           early_stopping_mult_factor: float):
        """Checks early stopping condition.

        In current version it raises `StopIteration` if current batch loss is greater than
        best smoothed batch loss in `early_stopping_mult_factor` times.

        Args:
            current_loss: loss value of current batch

        """
        if current_loss > (early_stopping_mult_factor * self.best_avg_loss):
            raise StopIteration

    def find_optimal_lr_borders(self, min_left_seq_len: int = 5,
                                min_right_seq_len: int = 2,
                                min_decreasing_gain: float = 0.999,
                                min_increasing_gain: float = 1,
                                **_ignored) -> Tuple[float, float]:
        """Finds left and right borders of learning rate where loss decreasing.

        Args:
            min_left_seq_len: minimum number of loss decreasing events in a row
                to define left lr border
            min_right_seq_len: minimum number of loss increasing events in a row
                to define right lr border
            min_decreasing_gain: minimum ratio of current loss to previous loss which defined
                as a loss decreasing event
            min_increasing_gain: minimum ratio of current loss to previous loss which defined
                as a loss increasing event
            **_ignored: ignored kwargs storage

        Returns:
            Tuple with two min and max learning rates
        """
        if len(self.logs['lr']) == 0:
            raise Exception('You must run range_test before')

        min_optimal_lr_idx = self._find_left_lr_border_idx(self.logs['avg_loss'],
                                                           min_decreasing_gain,
                                                           min_left_seq_len)
        max_optimal_lr_idx = self._find_right_lr_border_idx(self.logs['avg_loss'],
                                                            min_increasing_gain,
                                                            min_right_seq_len,
                                                            min_optimal_lr_idx)

        self.min_optimal_lr = self.logs['lr'][min_optimal_lr_idx]
        self.max_optimal_lr = self.logs['lr'][max_optimal_lr_idx]

        return self.min_optimal_lr, self.max_optimal_lr

    @classmethod
    def _find_left_lr_border_idx(cls, avg_losses: np.ndarray,
                                 min_decreasing_gain: float,
                                 min_decreasing_sequence_len: int) -> int:
        """Finds left border of optimal learning rate range

        Args:
            avg_losses: smoothed loss array
            min_decreasing_gain: minimum ratio of current loss to previous loss which defined
                as a loss decreasing event
            min_decreasing_sequence_len: minimum number of loss decreasing events in a row
                to define left lr border

        Returns:
            Left border idx of optimal learning rate
        """
        if min_decreasing_gain >= 1:
            raise ValueError('Param "min_decreasing_gain" must be strictly less than 1')
        if min_decreasing_sequence_len < 1:
            raise ValueError('Param "min_decreasing_sequence_len" must be strictly more than 0')

        avg_loss_gains = avg_losses[1:] / avg_losses[:-1]
        gains_less_threshold_idxs = (avg_loss_gains <= min_decreasing_gain).nonzero()[0]

        left_border_idx = cls.__find_monotony(gains_less_threshold_idxs, min_decreasing_sequence_len)

        if left_border_idx is None:
            raise Exception('Left learning rate border was not found')
        else:
            left_border_idx += 1  # add one to choose first lr after loss starts decreasing
            return left_border_idx

    @classmethod
    def _find_right_lr_border_idx(cls, avg_losses: np.ndarray,
                                  min_increasing_gain: float,
                                  min_increasing_sequence_len: int,
                                  left_border_idx: int) -> int:
        """Finds right border of optimal learning rate range

        Args:
            avg_losses: smoothed loss array
            min_increasing_gain: minimum ratio of current loss to previous loss which defined
                as a loss increasing event
            min_increasing_sequence_len: minimum number of loss increasing events in a row
                to define right lr border
            left_border_idx: idx of left border of lr in logs

        Returns:
            Right border idx of optimal learning rate
        """
        if min_increasing_gain < 1:
            raise ValueError('Param "min_increasing_gain" must be more or equal than 1')
        if min_increasing_sequence_len < 1:
            raise ValueError('Param "min_increasing_sequence_len" must be strictly more than 0')

        avg_losses_after_left_border = avg_losses[(left_border_idx+1):]
        avg_loss_gains = avg_losses_after_left_border[1:] / avg_losses_after_left_border[:-1]
        gains_more_threshold_idxs = (avg_loss_gains >= min_increasing_gain).nonzero()[0]

        right_border_idx = cls.__find_monotony(gains_more_threshold_idxs, min_increasing_sequence_len)

        if right_border_idx is None:
            raise Exception('Right learning rate border was not found')
        else:
            right_border_idx += (left_border_idx + 1)  # add one because of indexing starts from 0
            return right_border_idx

    @staticmethod
    def __find_monotony(gains_over_threshold_idxs: np.ndarray,
                        min_sequence_len: int) -> int:
        """Finds index of element from which loss began to change monotonically

        Args:
            gains_over_threshold_idxs: array with indices where ratio of current loss to
                previous loss more or less than chosen threshold
            min_sequence_len: minimum sequence of indices to define a monotony

        Returns:
            Index of loss array (under the `gains_over_threshold_idxs`) where monotony starts
        """
        border_idx = None
        sequence_len = 1
        for cur_idx, next_idx in zip(gains_over_threshold_idxs[:-1], gains_over_threshold_idxs[1:]):
            if sequence_len >= min_sequence_len:
                border_gain_idx = cur_idx - min_sequence_len
                border_idx = border_gain_idx + 1  # add one because of gains are shifted by one
                break

            if next_idx - cur_idx == 1:
                sequence_len += 1
            else:
                sequence_len = 1

        return border_idx

    def plot(self, out_mode: str,
             logs_path: Optional[Path] = None,
             **_ignored):
        """Plots figure with lr range test results (axes: lr, loss)

        Args:
            out_mode: "save" or "show" lr-loss figure
            logs_path: path to save plot
            **_ignored: ignored kwargs storage
        """
        if len(self.logs['lr']) == 0:
            raise Exception('You must run range_test before')

        loss, avg_loss, lr = [self.logs[i] for i in ('loss', 'avg_loss', 'lr')]

        # plot loss-lr figure
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.semilogx(lr, avg_loss, '-', label='avg loss')
        plt.semilogx(lr, loss, '--', label='raw loss')
        plt.grid(True, color='0.85')
        ax.set(title=self.model_name, ylabel='loss', xlabel='lr',
               xlim=[min(self.lrs), max(self.lrs)], ylim=[None, self.logs['avg_loss'][0] * 2])

        # plot optimal min/max lrs dots
        min_max_optimal_lr = (self.min_optimal_lr, self.max_optimal_lr)
        if None not in min_max_optimal_lr:
            borders_idxs = np.isin(lr, min_max_optimal_lr).nonzero()[0]
            plt.semilogx(lr[borders_idxs], avg_loss[borders_idxs], 'o')
            plt.vlines(lr[borders_idxs], *ax.get_ylim(), colors='r', linestyles='dashed')

        if out_mode == 'show':
            plt.show()
        elif out_mode == 'save':
            if logs_path is None:
                raise ValueError('Param "logs_path" must be given to save figure')

            plt.savefig(Path(logs_path, '{}.png'.format(self.model_name)))
        else:
            raise Exception('Param "out_mode" must be "show" or "save"')

    def run(self, **kwargs) -> Tuple[float, float]:
        """
        Run range test, plot lr-loss figure, try to find optimal lr range, update lr-loss figure

        Args:
            **kwargs: dict, which sent into kwargs in all used LRFinder methods

        Returns:
            Tuple with two min and max learning rates
        """
        self.range_test(**kwargs)
        self.plot(**kwargs)

        min_lr, max_lr = self.find_optimal_lr_borders(**kwargs)
        self.plot(**kwargs)

        return min_lr, max_lr


class ContinuousIterDataLoader:
    """
    A wrapper for continuous iterating over given `data_loader`.
    If `StopIteration` exception is raised than DataLoader will be automatically reseted.

    Args:
        data_loader: DataLoader to continuously iterate over it
    """
    def __init__(self, data_loader: 'DataLoader'):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self) -> 'Tensor':
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)

        return data
