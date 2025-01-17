import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Tuple

class SmoothKLDivLoss(nn.Module):
    """
    KLDivLoss with Label Smoothing.
    """
    def __init__(self, size: int, pad_token_idx: int, label_smoothing: float = 0.0, reduction: str = 'sum') -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.pad_token_idx = pad_token_idx
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert x.size(1) == self.size

        true_dist = x.data.clone().detach()
        true_dist.fill_(self.label_smoothing / (self.size-2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_token_idx] = 0

        mask = torch.nonzero(target.data == self.pad_token_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def compute_loss(criterion: Callable, model_out: Tensor, tgt: Tensor, norm: int|float, prediction_head: Callable, apply_softmax: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Calculates the loss using criterion for the output of a model and ground truth labels.

    Args:
    - `criterion`: Loss function to calculate the error.
    - `model_out`: Tensor with output from the model (commonly raw output before prediction_head).
    - `tgt`: Ground truth target tensor.
    - `norm`: Scalar value to normalize or scale the loss.
    - `prediction_head`: Function or layer applied to `model_out` before calculating the loss.
    - `apply_softmax`: Whether softmax should be applied in the prediction head.

    Returns:
    - The unnormalized loss.
    - The normalized loss.
    """
    if prediction_head is not None:
        out = prediction_head(model_out, apply_softmax)

    sloss = (
        criterion(
            out.contiguous().view(-1, out.size(-1)),
            tgt.contiguous().view(-1)
        )
        / norm
    )
    return sloss * norm, sloss


class LossCalculator:
    """
    Wrapper class to calculate the loss with a specified criterion and prediction head.
    """
    def __init__(self, criterion: Callable, prediction_head: Callable, apply_softmax: bool = True) -> None:
        """
        Args:
        - `criterion`: Loss function to calculate the error.
        - `prediction_head`: Function or layer applied to `model_out` before calculating the loss.
        - `apply_softmax`: Whether softmax should be applied in the prediction head.
        """
        self.criterion = criterion
        self.prediction_head = prediction_head
        self.apply_softmax = apply_softmax
    
    def __call__(self, model_out: Tensor, tgt: Tensor, norm: int|float) -> Tuple[Tensor, Tensor]:
        """
        Args:
        - `model_out`: Tensor with output from the model (commonly raw output before prediction_head).
        - `tgt`: Ground truth target tensor.
        - `norm`: Scalar value to normalize or scale the loss.

        Returns:
        - The unnormalized loss.
        - The normalized loss.
        """
        return compute_loss(self.criterion, model_out, tgt, norm, self.prediction_head, self.apply_softmax)
