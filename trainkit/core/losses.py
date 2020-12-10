import torch
import torch.nn as nn
import torch.nn.functional as torf


class FocalLoss(nn.Module):
    def __init__(self, gamma: float,
                 reduction: str,
                 eps: float = 1e-8,
                 log_loss_weight: torch.Tensor = None):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.register_buffer('nll_weight', log_loss_weight)

        self.nll_loss = nn.NLLLoss(weight=self.nll_weight, reduction='none')

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
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

        # there is no `alpha` because of there is `focal_weight`
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
