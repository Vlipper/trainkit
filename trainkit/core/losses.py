from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as torf

if TYPE_CHECKING:
    from typing import Literal, Optional
    from torch import Tensor


class FocalCELoss(nn.Module):
    """Focal loss (without parameter alpha)
    """

    def __init__(self, gamma: float,
                 reduction: 'Literal["none", "mean", "sum"]' = 'mean',
                 eps: float = 1e-8,
                 log_loss_weight: 'Optional[Tensor]' = None):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.register_buffer('nll_weight', log_loss_weight)
        self.nll_weight: 'Optional[Tensor]'

        self.nll_loss = nn.NLLLoss(weight=self.nll_weight, reduction='none')

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError('Param reduction must be on of: "none", "mean", "sum". '
                             f'Given: {reduction}')

    def forward(self, input: 'Tensor',
                target: 'Tensor') -> 'Tensor':
        """Calculates focal loss

        Args:
            input: unnormalized logits tensor with shape (N, C)
            target: targets tensor with shape (N,). Where `0 <= target[i] < num_classes`

        Returns:
            Focal loss scalar if `reduction` is not 'none' or vector (N,) otherwise
        """
        preds_soft = input.softmax(dim=1) + self.eps
        target_oh_mask = torf.one_hot(target, num_classes=input.shape[1])  # shape=(N, C)

        preds_target = (preds_soft * target_oh_mask).max(dim=1)[0]
        focal_weight = torch.pow(1 - preds_target, self.gamma)

        # ToDo: add `alpha` multiplier
        loss_tmp = focal_weight * self.nll_loss(preds_soft.log(), target)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            preds_weights = (target_oh_mask * self.nll_weight).max(dim=1)[0]
            loss = loss_tmp.sum() / preds_weights.sum()
        elif self.reduction == 'sum':
            loss = loss_tmp.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss


class FocalBCELoss(nn.Module):
    """Focal loss with BCE under hood
    """

    pos_weight: 'Optional[Tensor]'

    def __init__(self, gamma: float,
                 pos_weight: 'Optional[Tensor]' = None,
                 reduction: 'Literal["none", "mean", "sum"]' = 'mean'):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError('Param reduction must be on of: "none", "mean", "sum". '
                             f'Given: {reduction}')

    def forward(self, input: 'Tensor',
                target: 'Tensor') -> 'Tensor':
        """Calculates focal loss with BCE under hood

        Args:
            input: (N, C), unnormalized logits
            target: (N, C), target probabilities

        Returns:
            Scalar if `reduction` is not 'none' or matrix (N, C) otherwise
        """
        pos_preds = input.sigmoid()
        neg_preds = 1 - pos_preds

        # `pos_weight` here plays role of `alpha` from initial focal loss formula
        loss = -1 * (self.pos_weight * target * (neg_preds ** self.gamma) * pos_preds.log()
                     + (1 - target) * (pos_preds ** self.gamma) * neg_preds.log())

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss


class SmoothedBCEWithLogitsLoss(nn.Module):
    """Label smoothing loss with BCE under hood
    """

    def __init__(self, smooth_alpha: float,
                 smooth_multiplier: float = 0.5,
                 reduction: str = 'mean',
                 pos_weight: 'Optional[Tensor]' = None):
        super().__init__()

        if not (0 <= smooth_alpha <= 1):
            raise ValueError(f"Param 'smooth_alpha' must be between 0 and 1. Given: {smooth_alpha}")

        self.alpha = smooth_alpha
        self.smooth_multiplier = smooth_multiplier

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, input: 'Tensor',
                target: 'Tensor') -> 'Tensor':
        """Calculates BCE loss with label smoothing

        Args:
            input: (N, C), unnormalized logits
            target: (N, C), target probabilities

        Returns:
            Scalar if `reduction` is not 'none' or tensor (N, C) otherwise
        """
        smoothed_target = target * (1 - self.alpha) + self.smooth_multiplier * self.alpha
        loss = self.bce_loss_fn.forward(input, smoothed_target)

        return loss


class FloodingLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module,
                 flooding_level: float):
        super().__init__()

        self.loss_fn = loss_fn
        self.fl = flooding_level

    def forward(self, *args, **kwargs) -> 'Tensor':
        loss = self.loss_fn.forward(*args, **kwargs)
        loss = (loss - self.fl).abs() + self.fl

        return loss
