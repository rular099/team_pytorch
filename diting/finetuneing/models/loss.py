import torch.nn as nn
import torch
from torch.nn import HuberLoss
from typing import Tuple

class CELoss(nn.Module):
    """
    Cross Entropy loss for dpk task
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L) or (N,Classes)"""
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= self.weight
        if loss.ndim > 1:
            loss = loss.sum(1).mean()
        else:
            loss = loss.mean()
        return loss

class BazLoss(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = HuberLoss()
    def forward(self, preds, targets):
        # to radians
        angles_radians = torch.deg2rad(targets)
        # cos and sin for target
        cos_values = torch.cos(angles_radians)
        sin_values = torch.sin(angles_radians)
        loss = 0.5 * self.loss_fn(preds[0], cos_values) + 0.5 * self.loss_fn(preds[1], sin_values)
        return loss

class BCELoss(nn.Module):
    """
    Binary cross entropy loss for phase-picking and detection
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weight:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets,show_loss=False):
        """Input shape: (N,C,L)"""
        loss = -(
            targets * torch.log(preds + self._epsilon)
            + (1 - targets) * torch.log(1 - preds + self._epsilon)
        )
        loss_ = self.weight * loss
        loss = loss_.mean()
        if show_loss:
            return loss,loss_
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss
    """

    _epsilon = 1e-6

    def __init__(self, gamma=2, weight=None, has_softmax=True):
        """
        Args:
            gamma (float): Coefficient.
            weight (list|Tensor): Weight of each class. Defaults to None.
            has_softmax (bool): If True, softmax will be applied for the input `preds`. Defaults to True.
        """
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

        self.has_softmax = has_softmax

    def forward(self, preds, targets):
        """Input shape: (N,C,L) or (N,Classes)"""
        if self.has_softmax:
            preds = torch.nn.functional.softmax(preds, dim=1)
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= torch.pow((1 - preds), self.gamma)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Focal loss (binary)

    note: the input `preds` must be the output of `sigmoid`.
    """

    _epsilon = 1e-6

    def __init__(self, gamma=2, alpha=1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""

        loss = -(
            self.alpha
            * torch.pow((1 - preds), self.gamma)
            * targets
            * torch.log(preds + self._epsilon)
            + (1 - self.alpha)
            * torch.pow(preds, self.gamma)
            * (1 - targets)
            * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss


class CircleLoss(nn.Module):
    """
    Circle loss (binary)

    note: the input `preds` must be the output of `sigmoid`.
    """

    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32).transpose(0, 1)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""
        preds = (1 - 2 * targets) * preds
        y_pred_neg = preds - targets * 1e12
        y_pred_pos = preds - (1 - targets) * 1e12
        zeros = torch.zeros_like(preds[..., :1], device=preds.device)
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        loss *= self.weight
        loss = loss.mean()
        return loss

      
class MSELoss(nn.Module):
    """
    MSE Loss.
    """

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""
        loss = (preds - targets) ** 2
        loss *= self.weight
        loss = loss.mean()
        return loss


class CombinationLoss(nn.Module):
    """
    For multi-task learning.
    """

    def __init__(self, losses: list, losses_weights: list = None) -> None:
        """
        note: Use `functools.partial` if there are arguments that need to be passed to the loss module in `losses`.
        """
        super().__init__()

        assert len(losses) > 0

        if len(losses) == 1:
            raise Exception(
                f"Expected number of losses `>=2`, got {len(losses)}."
                f" `CombinationLoss` is used for multi-task training, and requires at least two loss modules."
                f" Use `{losses[0]}` instead."
            )

        if losses_weights is not None:
            assert len(losses) == len(losses_weights)
            self.losses_weights = losses_weights
        else:
            self.losses_weights = [1.0] * len(losses)

        self.losses = nn.ModuleList([Loss() for Loss in losses])

    def forward(self, preds: Tuple[torch.Tensor], targets: Tuple[torch.Tensor]):
        sum_loss = 0.0
        for i, (pred, target, lossfn, weight) in enumerate(
            zip(preds, targets, self.losses, self.losses_weights)
        ):
            sum_loss += lossfn(pred, target) * weight

        return sum_loss


class MousaviLoss(nn.Module):
    """
    Loss module for the following models:
    
        [1] MagNet. Mousavi et al. 2019
        [2] dist-PT Network. Mousavi et al. 2020
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        y_hat = preds[:, 0].reshape(-1, 1)
        s = preds[:, 1].reshape(-1, 1)
        loss = torch.sum(
            0.5 * torch.exp(-1 * s) * torch.square(torch.abs(targets - y_hat)) + 0.5 * s
        )
        return loss

class LossConfig:
    loss_fns = {'det':BCELoss(),
                'ppk':BCELoss(),
                'spk':BCELoss(),
                'dis':HuberLoss(),
                'emg':HuberLoss()}

def multi_task_loss_fn(outputs, targets, metrics_targets, model_tasks, task_loss_weight,
                       meta_data=None, fused_feature=False, hed_loss_weight=None):
    loss = 0
    loss_log_dict = {}
    assert len(model_tasks) == len(task_loss_weight)
        
    for id,(task,weight) in enumerate(zip(model_tasks,task_loss_weight)):
        if task == 'det':
            valid_indices = (metrics_targets[task] != torch.tensor([-1,-1])).any(dim=1)
        else:
            valid_indices = (metrics_targets[task] != -1).squeeze()
        if fused_feature:
            output_for_loss = []
            for item in outputs[task]:
                output_for_loss.append(item[valid_indices])
        else:
#            print("metrics_targets: ",metrics_targets)
#            print("targets: ",targets[task].shape,targets)
#            print("outputs: ",outputs[task].shape,outputs, "valid_indices: ",valid_indices)
            output_for_loss = outputs[task][valid_indices]
        target_for_loss = targets[task][valid_indices]
#        print("target_for_loss_shape: ",target_for_loss.shape,target_for_loss)
#        print("output_for_loss_shape: ",output_for_loss.shape,output_for_loss)

        if output_for_loss.numel() == 0:
            continue
        
        loss_fn = LossConfig.loss_fns[task]
#        if task in ['det','ppk','spk']:
#            loss_fn = BCELoss()
#        elif task in ['dis','emg']:
#            loss_fn = HuberLoss()
            
        # added by lgy 
        if fused_feature:
            sub_loss = 0
            for i in range(len(outputs)):
                if i == 0:
                    continue
                sub_loss += hed_loss_weight[i - 1] * loss_fn(output_for_loss[i], target_for_loss) 
        else:
            sub_loss = loss_fn(output_for_loss,target_for_loss)
        
        loss += weight * sub_loss
        loss_log_dict[task] = sub_loss.item()

    return loss,loss_log_dict

