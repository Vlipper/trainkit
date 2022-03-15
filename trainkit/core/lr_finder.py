import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Dict, Tuple, Union

    from torch import Tensor
    from torch.utils.data import DataLoader
    from trainkit.core.trainer import Trainer

_logger = logging.getLogger(__name__)


class LRFinder:
    """LRFinder

    Args:
        trainer: Trainer instance
        min_lr: minimum learning rate for range test
        max_lr: maximum learning rate for range test
        num_lrs: number of learning rates to check
        **_ignored: ignored kwargs storage
    """
    logs: 'Dict[str, Union[list, np.ndarray]]'

    def __init__(self, trainer: 'Trainer',
                 min_lr: float = 1e-7,
                 max_lr: float = 0.9,
                 num_lrs: int = 200,
                 **_ignored):
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.model_name = trainer.model_name
        self.data_loader = ContinuousDataLoader(trainer.train_loader)
        self.tb_writer = trainer.log_writer.tb_writer

        # vars for range_test
        self.lrs = np.geomspace(start=min_lr, stop=max_lr, num=num_lrs)
        self.logs = {'lr': [], 'loss': [], 'smooth_loss': []}
        self.min_optimal_lr, self.max_optimal_lr = None, None
        self.best_smooth_loss = None

    def range_test(self, smooth_beta: float = 0.8,
                   check_early_stopping: bool = True,
                   early_stopping_mult_factor: float = 3,
                   **_ignored):
        """Runs learning rate range test.

        Trains model on data from given `DataLoader` with applying new learning rate before every
        batch. After every batch step it calculates smoothed loss and stores logs into `self.logs`.
        Smoothed loss formula looks like this:
        `smooth_beta * smooth_loss + (1 - smooth_beta) * batch_loss`.

        Args:
            smooth_beta: coefficient at previous value of smooth loss
            check_early_stopping: checks loss explosion and necessity of training break
            early_stopping_mult_factor: multiplier value, can be explained as "in how many times
                current batch loss must be bigger than best smoothed loss to raise `StopIteration`"
            **_ignored: ignored kwargs storage
        """
        smooth_loss = None
        self.model.train()

        for iter_step, lr in enumerate(tqdm(self.lrs, desc='finding LR')):
            # apply new lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            _logger.info('Next LR applied. Step: %d | lr: %.2e', iter_step, lr)

            # train step
            batch = next(self.data_loader)
            batch_loss = self.trainer.optim_wrapper(self.model.batch_step)(iter_step, batch)

            # calculate smoothed loss
            if iter_step == 0:
                smooth_loss = batch_loss
            else:
                smooth_loss = smooth_beta * smooth_loss + (1 - smooth_beta) * batch_loss

            # store logs
            self.logs['lr'].append(lr)
            self.logs['loss'].append(batch_loss)
            self.logs['smooth_loss'].append(smooth_loss)
            self.tb_writer.add_scalar('lr-rt/lr', lr, iter_step)
            self.tb_writer.add_scalar('lr-rt/loss', batch_loss, iter_step)
            self.tb_writer.add_scalar('lr-rt/smooth-loss', smooth_loss, iter_step)

            if check_early_stopping:
                if self.best_smooth_loss is None or self.best_smooth_loss > smooth_loss:
                    self.best_smooth_loss = smooth_loss

                try:
                    self.__early_stopping_check(batch_loss, early_stopping_mult_factor)
                except StopIteration:
                    _logger.info('LR range test was early stopped at lr: %.2e', lr)
                    break

        # add lr-rt scalars on one layout
        layout = {'lr-rt': {'loss': ['Multiline', ['lr-rt/loss', 'lr-rt/lr']],
                            'smooth-loss': ['Multiline', ['lr-rt/smooth-loss', 'lr-rt/lr']]}}
        self.tb_writer.add_custom_scalars(layout)

        self.tb_writer.flush()
        self.logs.update({key: np.array(val) for key, val in self.logs.items()})
        self.trainer.rollback_states()

    def __early_stopping_check(self, current_loss: float,
                               early_stopping_mult_factor: float):
        """Checks early stopping condition.

        Args:
            current_loss: loss value of current batch
            early_stopping_mult_factor: multiplicative factor of best smooth loss when it compared
                with current batch loss

        Raises:
            StopIteration: Raised when current batch loss is greater than best smoothed batch loss
                in `early_stopping_mult_factor` times.
        """
        if current_loss > (early_stopping_mult_factor * self.best_smooth_loss):
            raise StopIteration

    def _find_optimal_lr_borders(self, min_left_seq_len: int = 5,
                                 min_right_seq_len: int = 2,
                                 min_decreasing_gain: float = 0.999,
                                 min_increasing_gain: float = 1,
                                 **_ignored) -> 'Tuple[float, float]':
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
            raise Exception('You must run range test before: range_test()')

        min_optimal_lr_idx = self._find_left_lr_border_idx(self.logs['smooth_loss'],
                                                           min_decreasing_gain,
                                                           min_left_seq_len)
        max_optimal_lr_idx = self._find_right_lr_border_idx(self.logs['smooth_loss'],
                                                            min_increasing_gain,
                                                            min_right_seq_len,
                                                            min_optimal_lr_idx)

        self.min_optimal_lr = self.logs['lr'][min_optimal_lr_idx]
        self.max_optimal_lr = self.logs['lr'][max_optimal_lr_idx]
        _logger.info('Optimal LR border was found. Selected min/max LRs: %.2e | %.2e',
                     self.min_optimal_lr, self.max_optimal_lr)

        return self.min_optimal_lr, self.max_optimal_lr

    def _find_left_lr_border_idx(self, smooth_losses: np.ndarray,
                                 min_decreasing_gain: float,
                                 min_decreasing_sequence_len: int) -> int:
        """Finds left border of optimal learning rate range.

        Args:
            smooth_losses: smoothed loss array
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

        smooth_loss_gains = smooth_losses[1:] / smooth_losses[:-1]
        gains_less_threshold_idxs = (smooth_loss_gains <= min_decreasing_gain).nonzero()[0]

        left_border_idx = self.__find_monotony(gains_less_threshold_idxs,
                                               min_decreasing_sequence_len)
        if left_border_idx is None:
            raise Exception('Left learning rate border was not found')

        left_border_idx += 1  # add one to choose first lr after loss starts decreasing
        return left_border_idx

    def _find_right_lr_border_idx(self, smooth_losses: np.ndarray,
                                  min_increasing_gain: float,
                                  min_increasing_sequence_len: int,
                                  left_border_idx: int) -> int:
        """Finds right border of optimal learning rate range.

        Args:
            smooth_losses: smoothed loss array
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

        smooth_losses_after_left_border = smooth_losses[(left_border_idx + 1):]
        smooth_loss_gains = smooth_losses_after_left_border[1:] \
            / smooth_losses_after_left_border[:-1]
        gains_more_threshold_idxs = (smooth_loss_gains >= min_increasing_gain).nonzero()[0]

        right_border_idx = self.__find_monotony(gains_more_threshold_idxs,
                                                min_increasing_sequence_len)
        if right_border_idx is None:
            raise Exception('Right learning rate border was not found')

        right_border_idx += (left_border_idx + 1)  # add one because of indexing starts from 0
        return right_border_idx

    @staticmethod
    def __find_monotony(gains_over_threshold_idxs: np.ndarray,
                        min_sequence_len: int) -> int:
        """Finds index of element from which loss began to change monotonically.

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

    @staticmethod
    def _find_optimal_lr(min_lr: float,
                         max_lr: float) -> float:
        """Calculates optimal learning rate between given min and max borders.

        Args:
            min_lr: minimal LR border
            max_lr: maximum LR border

        Returns:
            'Optimal' LR between min and max borders
        """
        optimal_lr = min_lr + 0.5 * (max_lr - min_lr)
        _logger.info('Optimal LR was selected as %.2e', optimal_lr)

        return optimal_lr

    def __call__(self, **kwargs) -> 'Tuple[float, float, float]':
        """Runs LR range test and tries to find borders of optimal lr.

        Args:
            **kwargs: dict, which used as kwargs in all LRFinder methods

        Returns:
            Min/max borders of optimal lr and one median value of optimal lr
        """
        self.range_test(**kwargs)
        min_lr, max_lr = self._find_optimal_lr_borders(**kwargs)
        optimal_lr = self._find_optimal_lr(min_lr, max_lr)

        return min_lr, max_lr, optimal_lr


class ContinuousDataLoader:
    """A wrapper which continuously iterates over given `DataLoader` instance.

    It iterates until `StopIteration` exception raised. After that wrapper reset given `DataLoader`
    and continue iteration.

    Args:
        data_loader: `DataLoader` instance
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
