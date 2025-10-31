"""
Implementation of loss functions.
ref: https://github.com/senli1073/SeisT/blob/main/models/loss.py
"""


import torch.nn as nn
import torch
from torch.nn import HuberLoss
import torch.nn.functional as F
from typing import Tuple

class PmpClsLoss(nn.Module):
    """
    Cross entropy loss for pmp classification task
    """

    def __init__(self, weight=[1.0, 1.0]) -> None:
        super().__init__()

        if weight is not None:
            print(f"[{self._get_name()}] Loss Weight:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

        assert weight.shape == (2,)

    def forward(self, preds, targets):
        """Input shape: (N, 2)"""
        loss = F.cross_entropy(preds, targets)
        loss *= self.weight
        loss = loss.mean()
        return loss


class DPKLoss(nn.Module):
    """
    Binary cross entropy loss for phase-picking and detection task
    """

    _epsilon = 1e-6

    def __init__(self, weight=[[0.5], [1.0], [1.0]]) -> None:
        super().__init__()

        if weight is not None:
            print(f"[{self._get_name()}] Loss Weight:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

        assert weight.shape == (3, 1)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""

        # logits -> sigmoid by 2024.1.22 wnzz
        preds = F.sigmoid(preds)

        loss = -(
            targets * torch.log(preds + self._epsilon)
            + (1 - targets) * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss