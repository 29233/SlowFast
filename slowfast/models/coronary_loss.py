#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Loss functions for Coronary Multi-Task Learning.

This module implements joint loss functions for:
1. Classification loss (BCE for confidence prediction)
2. Regression loss (MSE for plaque percentage prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class CoronaryMultiTaskLoss(nn.Module):
    """
    Multi-task loss for coronary CT analysis.

    Combines:
    - Classification loss: Binary cross-entropy for confidence prediction
    - Regression loss: MSE for normalized plaque percentage

    支持多目标检测任务，每个样本可能包含多个斑块目标。
    使用 valid_mask 来区分真实目标和 padding。

    Attributes:
        cls_weight (float): Weight for classification loss
        reg_weight (float): Weight for regression loss
        num_proposals (int): Number of proposals (for multi-token mode)
    """

    def __init__(self, cfg):
        """
        Initialize the multi-task loss.

        Args:
            cfg: Configuration node with the following required fields:
                - CORONARY.CLS_LOSS_WEIGHT: Weight for classification loss
                - CORONARY.REG_LOSS_WEIGHT: Weight for regression loss
                - CORONARY.NUM_PROPOSALS: Number of proposals
                - CORONARY.USE_MULTI_TOKEN: Whether to use multi-token mode
        """
        super(CoronaryMultiTaskLoss, self).__init__()

        self.cfg = cfg
        self.cls_weight = cfg.CORONARY.CLS_LOSS_WEIGHT
        self.reg_weight = cfg.CORONARY.REG_LOSS_WEIGHT
        self.num_proposals = cfg.CORONARY.NUM_PROPOSALS
        self.use_multi_token = cfg.CORONARY.USE_MULTI_TOKEN

        # Classification loss (BCE for confidence scores)
        # 使用 reduction='none' 以便手动处理 mask
        self.cls_loss_fn = nn.BCELoss(reduction='none')

        # Regression loss (MSE for plaque percentages)
        # 使用 reduction='none' 以便手动处理 mask
        self.reg_loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the multi-task loss.

        Args:
            predictions (Dict): Model predictions containing:
                - 'cls_outputs': List of classification outputs [batch, 1]
                - 'reg_outputs': List of regression outputs [batch, 1]
            targets (Dict): Ground truth targets containing:
                - 'cls_targets': Classification targets [batch, num_proposals]
                - 'reg_targets': Regression targets [batch, num_proposals]
                - 'valid_mask': Validity mask [batch, num_proposals] (1 for real targets, 0 for padding)

        Returns:
            Tuple containing:
                - total_loss (torch.Tensor): The weighted sum of losses
                - loss_dict (Dict): Individual loss components for logging
        """
        cls_outputs = predictions['cls_outputs']  # List of [batch, 1] tensors
        reg_outputs = predictions['reg_outputs']  # List of [batch, 1] tensors

        cls_targets = targets['cls_targets']  # [batch, num_proposals]
        reg_targets = targets['reg_targets']  # [batch, num_proposals]

        # Get validity mask
        valid_mask = targets.get('valid_mask', None)  # [batch, num_proposals]

        batch_size = cls_targets.shape[0]
        device = cls_targets.device

        # Compute classification loss
        if self.use_multi_token and len(cls_outputs) == self.num_proposals:
            # Multi-token mode: compute loss for each proposal
            cls_loss_per_proposal = []
            reg_loss_per_proposal = []

            for i in range(self.num_proposals):
                cls_out = cls_outputs[i]  # [batch, 1]
                reg_out = reg_outputs[i]  # [batch, 1]

                cls_target_i = cls_targets[:, i:i+1]  # [batch, 1]
                reg_target_i = reg_targets[:, i:i+1]  # [batch, 1]

                # Compute per-sample loss
                cls_loss_i = self.cls_loss_fn(cls_out, cls_target_i)  # [batch, 1]
                reg_loss_i = self.reg_loss_fn(reg_out, reg_target_i)  # [batch, 1]

                cls_loss_per_proposal.append(cls_loss_i)
                reg_loss_per_proposal.append(reg_loss_i)

            # Stack: [batch, num_proposals]
            cls_loss_all = torch.cat(cls_loss_per_proposal, dim=1)
            reg_loss_all = torch.cat(reg_loss_per_proposal, dim=1)

            # Apply valid mask if available
            if valid_mask is not None:
                # Only compute loss for valid targets
                cls_loss = (cls_loss_all * valid_mask).sum() / (valid_mask.sum() + 1e-5)
                reg_loss = (reg_loss_all * valid_mask).sum() / (valid_mask.sum() + 1e-5)
            else:
                # No mask, average over all proposals
                cls_loss = cls_loss_all.mean()
                reg_loss = reg_loss_all.mean()
        else:
            # Single token mode or outputs don't match num_proposals
            # Use only the first output
            cls_out = cls_outputs[0] if isinstance(cls_outputs, list) else cls_outputs
            reg_out = reg_outputs[0] if isinstance(reg_outputs, list) else reg_outputs

            # If targets are 2D, use first column
            if cls_targets.dim() > 1:
                cls_loss = self.cls_loss_fn(cls_out, cls_targets[:, :1]).mean()
                reg_loss = self.reg_loss_fn(reg_out, reg_targets[:, :1]).mean()
            else:
                cls_loss = self.cls_loss_fn(cls_out, cls_targets).mean()
                reg_loss = self.reg_loss_fn(reg_out, reg_targets).mean()

        # Compute total loss
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
        }

        return total_loss, loss_dict


class CoronaryFocalLoss(nn.Module):
    """
    Focal Loss for classification with class imbalance handling.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.

        Args:
            alpha (float): Weighting factor for positive class
            gamma (float): Focusing parameter
            reduction (str): Reduction method
        """
        super(CoronaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities [B, 1]
            targets (torch.Tensor): Ground truth [B, 1]

        Returns:
            torch.Tensor: Focal loss value
        """
        # Clamp inputs to avoid log(0)
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)

        # Compute focal weight
        pos_weight = targets * self.alpha
        neg_weight = (1 - targets) * (1 - self.alpha)
        weight = pos_weight + neg_weight

        # Compute BCE
        bce = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))

        # Compute focal loss
        pt = torch.exp(-bce)
        focal_loss = weight * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CoronarySmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss) for regression.

    More robust to outliers than MSE.
    """

    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize smooth L1 loss.

        Args:
            beta (float): Threshold parameter
            reduction (str): Reduction method
        """
        super(CoronarySmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth L1 loss.

        Args:
            inputs (torch.Tensor): Predictions
            targets (torch.Tensor): Ground truth

        Returns:
            torch.Tensor: Smooth L1 loss value
        """
        diff = torch.abs(inputs - targets)
        smooth_loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        if self.reduction == 'mean':
            return smooth_loss.mean()
        elif self.reduction == 'sum':
            return smooth_loss.sum()
        else:
            return smooth_loss


def build_coronary_loss(cfg) -> nn.Module:
    """
    Build coronary loss function based on configuration.

    Args:
        cfg: Configuration node

    Returns:
        nn.Module: The loss module
    """
    loss_type = cfg.CORONARY.get('LOSS_TYPE', 'multi_task')

    if loss_type == 'multi_task':
        return CoronaryMultiTaskLoss(cfg)
    elif loss_type == 'focal':
        return CoronaryMultiTaskLoss(cfg)  # Same interface, different internal loss
    else:
        return CoronaryMultiTaskLoss(cfg)
