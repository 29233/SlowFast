#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Hungarian Loss for Set-based Coronary CT Multi-task Learning with Multi-class Classification.

This module implements set-based prediction with Hungarian matching for:
1. Classification task: multi-class classification (4 branches + 1 background)
2. Regression task: normalized plaque percentage (0-1)

The key idea is to formulate the multi-label prediction as a set prediction problem,
where we find the optimal one-to-one matching between predictions and ground truth
targets using the Hungarian algorithm.

Reference: Hungarian matching strategy from DETR and similar set-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional


class HungarianMatcher(nn.Module):
    """
    Computes the optimal matching between predictions and ground truth targets
    using the Hungarian algorithm.

    For each prediction i and target j, the matching cost is:
        C(i, j) = -alpha * p_i(c_j) + beta * L_reg(v_i, v_j)

    where:
        - p_i(c_j) is the predicted probability for target class (after softmax)
        - L_reg is the regression loss (L2/MSE)
        - alpha, beta are weighting factors
    """

    def __init__(
        self,
        cost_cls_weight: float = 1.0,
        cost_reg_weight: float = 1.0,
        cost_reg_type: str = 'mse',
        num_classes: int = 5,  # 4 branches + 1 background
    ):
        """
        Initialize the Hungarian matcher.

        Args:
            cost_cls_weight (float): Weight for classification cost in matching
            cost_reg_weight (float): Weight for regression cost in matching
            cost_reg_type (str): Type of regression loss for cost ('mse' or 'l1')
            num_classes (int): Number of classification classes (default: 5 = 4 branches + background)
        """
        super().__init__()
        self.cost_cls_weight = cost_cls_weight
        self.cost_reg_weight = cost_reg_weight
        self.cost_reg_type = cost_reg_type
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(
        self,
        cls_outputs: List[torch.Tensor],
        reg_outputs: List[torch.Tensor],
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Compute the optimal assignment between predictions and targets.

        Args:
            cls_outputs: List of classification logits, each [batch, num_classes]
            reg_outputs: List of regression outputs, each [batch, 1]
            cls_targets: Ground truth classification targets [batch, num_proposals]
                        (class indices: 0-3 for branches, 4 for background/-1 for padding)
            reg_targets: Ground truth regression targets [batch, num_proposals]
            valid_mask: Mask indicating valid targets [batch, num_proposals]
                       (1 for real targets, 0 for padding/background)

        Returns:
            List of tuples (pred_indices, target_indices) for each sample in batch
        """
        batch_size = cls_targets.shape[0]
        num_proposals = len(cls_outputs)  # Number of prediction slots

        # Stack outputs
        # cls_outputs: List of [B, num_classes] -> [B, N, num_classes]
        cls_pred = torch.stack(cls_outputs, dim=1)  # [B, N, num_classes]
        cls_pred = F.softmax(cls_pred, dim=-1)  # Convert logits to probabilities
        reg_pred = torch.stack(reg_outputs, dim=1).squeeze(-1)  # [B, N]

        result = []

        for b in range(batch_size):
            # Get predictions for this sample
            cls_pred_b = cls_pred[b]  # [N, num_classes]
            reg_pred_b = reg_pred[b]  # [N]

            # Get targets for this sample
            cls_tgt_b = cls_targets[b]  # [N]
            reg_tgt_b = reg_targets[b]  # [N]

            # Get valid mask for this sample
            if valid_mask is not None:
                mask_b = valid_mask[b]  # [N]
            else:
                # Assume targets with class >= 0 are valid (not padding)
                mask_b = (cls_tgt_b >= 0).float()

            # Find indices of foreground targets (class 0-3, i.e., < num_classes - 1)
            # Foreground classes are 0-3 (LAD, RCA, LCX, LM)
            # Background class is 4 (num_classes - 1)
            # Padding is -1 (excluded by valid_mask or class >= 0 check)
            foreground_mask = (cls_tgt_b >= 0) & (cls_tgt_b < self.num_classes - 1)
            valid_target_indices = torch.where(foreground_mask)[0].cpu().tolist()
            num_valid_targets = len(valid_target_indices)

            if num_valid_targets == 0:
                # No valid foreground targets, all predictions should be background
                result.append(([], []))
                continue

            # Build cost matrix: [num_proposals, num_valid_targets]
            # We need to find optimal assignment between N predictions and M valid targets

            # Classification cost: negative probability for the target class
            # For each prediction i and target j: cost_cls = -cls_pred[i, target_class_j]
            target_classes = cls_tgt_b[valid_target_indices].long()  # [M]
            # Extract probability for target class for each prediction
            # cls_pred_b: [N, num_classes], target_classes: [M]
            # We want cls_pred_b[i, target_classes[j]] for all i, j -> [N, M]
            cost_cls = -cls_pred_b[:, target_classes]  # [N, M] - directly index columns

            # Regression cost: L2 distance between predictions and targets
            reg_pred_expanded = reg_pred_b.unsqueeze(1)  # [N, 1]
            reg_tgt_valid = reg_tgt_b[valid_target_indices]  # [M]

            if self.cost_reg_type == 'mse':
                cost_reg = (reg_pred_expanded - reg_tgt_valid.unsqueeze(0)) ** 2  # [N, M]
            elif self.cost_reg_type == 'l1':
                cost_reg = torch.abs(reg_pred_expanded - reg_tgt_valid.unsqueeze(0))  # [N, M]
            else:
                raise ValueError(f"Unknown reg cost type: {self.cost_reg_type}")

            # Total cost
            cost_matrix = self.cost_cls_weight * cost_cls + self.cost_reg_weight * cost_reg
            cost_matrix = cost_matrix.cpu().numpy()  # [N, M]

            # Hungarian algorithm: find optimal assignment
            # row_ind: prediction indices, col_ind: target indices (in valid_target_indices)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Convert target indices back to original indices
            target_indices = [valid_target_indices[c] for c in col_ind]
            pred_indices = row_ind.tolist()

            result.append((pred_indices, target_indices))

        return result


class HungarianLoss(nn.Module):
    """
    Set-based Hungarian loss for coronary multi-task learning with multi-class classification.

    After finding the optimal matching using Hungarian algorithm, compute:
        L = sum_i [ L_cls(c_i, p_i) + 1{c_i != background} * lambda_reg * L_reg(v_i, v_i_hat) ]

    For predictions matched to background (no target), only classification loss is computed.
    """

    def __init__(
        self,
        cfg,
    ):
        """
        Initialize the Hungarian loss.

        Args:
            cfg: Configuration node with:
                - CORONARY.NUM_PROPOSALS: Number of proposal tokens
                - CORONARY.NUM_BRANCHES: Number of branch classes (default: 4)
                - CORONARY.CLS_LOSS_WEIGHT: Weight for classification loss
                - CORONARY.REG_LOSS_WEIGHT: Weight for regression loss
                - HUNGARIAN: Hungarian matching parameters
        """
        super().__init__()

        self.cfg = cfg
        self.num_proposals = cfg.CORONARY.NUM_PROPOSALS
        self.num_branches = cfg.CORONARY.get('NUM_BRANCHES', 4)
        self.num_classes = self.num_branches + 1  # +1 for background
        self.cls_weight = cfg.CORONARY.CLS_LOSS_WEIGHT
        self.reg_weight = cfg.CORONARY.REG_LOSS_WEIGHT

        # Hungarian matcher
        hungarian_cfg = cfg.HUNGARIAN if hasattr(cfg, 'HUNGARIAN') else None
        cost_cls_weight = hungarian_cfg.COST_CLS_WEIGHT if hungarian_cfg else 1.0
        cost_reg_weight = hungarian_cfg.COST_REG_WEIGHT if hungarian_cfg else 1.0
        cost_reg_type = hungarian_cfg.COST_REG_TYPE if hungarian_cfg else 'mse'

        self.matcher = HungarianMatcher(
            cost_cls_weight=cost_cls_weight,
            cost_reg_weight=cost_reg_weight,
            cost_reg_type=cost_reg_type,
            num_classes=self.num_classes,
        )

        # Classification loss: CrossEntropyLoss for multi-class
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Regression loss (MSE with reduction='none' for manual control)
        self.reg_loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Hungarian loss.

        Args:
            predictions: Model predictions containing:
                - 'cls_outputs': List of classification logits [batch, num_classes]
                - 'reg_outputs': List of regression outputs [batch, 1]
            targets: Ground truth targets containing:
                - 'cls_targets': Classification targets [batch, num_proposals]
                                (class indices: 0-3 for branches, 4 for background, padded with 0)
                - 'reg_targets': Regression targets [batch, num_proposals]
                - 'valid_mask': Validity mask [batch, num_proposals] (1 for real targets, 0 for padding)

        Returns:
            Tuple containing:
                - total_loss: Weighted sum of losses
                - loss_dict: Individual loss components for logging
        """
        cls_outputs = predictions['cls_outputs']  # List of [B, num_classes]
        reg_outputs = predictions['reg_outputs']  # List of [B, 1]

        cls_targets = targets['cls_targets']  # [B, N]
        reg_targets = targets['reg_targets']  # [B, N]
        valid_mask = targets.get('valid_mask', None)  # [B, N]

        batch_size = cls_targets.shape[0]
        device = cls_targets.device

        # Step 1: Find optimal matching using Hungarian algorithm
        # Only match foreground targets (class 0-3)
        matches = self.matcher(
            cls_outputs=cls_outputs,
            reg_outputs=reg_outputs,
            cls_targets=cls_targets,
            reg_targets=reg_targets,
            valid_mask=valid_mask,
        )

        # Step 2: Compute loss based on matching
        cls_loss_all = []
        reg_loss_all = []
        num_positive = 0

        for b in range(batch_size):
            pred_indices, target_indices = matches[b]

            # Get valid mask for this sample
            if valid_mask is not None:
                mask_b = valid_mask[b]  # [N]
            else:
                mask_b = torch.ones(self.num_proposals, device=device)

            # Get predictions and targets for matched pairs (only foreground targets)
            if len(pred_indices) > 0:
                for pred_idx, tgt_idx in zip(pred_indices, target_indices):
                    # Only compute loss for valid foreground targets
                    if mask_b[tgt_idx] > 0:
                        # Classification loss
                        cls_pred = cls_outputs[pred_idx][b]  # [num_classes]
                        cls_tgt = cls_targets[b, tgt_idx].long()  # scalar (class index 0-3)
                        cls_loss = self.cls_loss_fn(cls_pred.unsqueeze(0), cls_tgt.unsqueeze(0))[0]
                        cls_loss_all.append(cls_loss)

                        # Regression loss (only for foreground targets, class 0-3)
                        reg_pred = reg_outputs[pred_idx][b, 0]  # scalar
                        reg_tgt = reg_targets[b, tgt_idx]  # scalar
                        reg_loss = self.reg_loss_fn(reg_pred, reg_tgt)
                        reg_loss_all.append(reg_loss)
                        num_positive += 1

            # For unmatched predictions: should predict background or ignore if matched to padding
            all_pred_indices = set(range(self.num_proposals))
            matched_pred_indices = set(pred_indices)
            unmatched_indices = list(all_pred_indices - matched_pred_indices)

            for pred_idx in unmatched_indices:
                # Only classification loss (predict background)
                cls_pred = cls_outputs[pred_idx][b]  # [num_classes]
                # Background class index
                bg_class = self.num_classes - 1
                cls_tgt = cls_targets.new_tensor(bg_class).long()
                cls_loss = self.cls_loss_fn(cls_pred.unsqueeze(0), cls_tgt.unsqueeze(0))[0]
                cls_loss_all.append(cls_loss)

        # Aggregate losses
        if len(cls_loss_all) > 0:
            cls_loss_all = torch.stack(cls_loss_all)
            cls_loss = cls_loss_all.mean()
        else:
            cls_loss = cls_targets.new_tensor(0.0)

        if len(reg_loss_all) > 0:
            reg_loss_all = torch.stack(reg_loss_all)
            reg_loss = reg_loss_all.mean()
        else:
            reg_loss = reg_targets.new_tensor(0.0)

        # Compute total loss
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'num_positive': torch.tensor(num_positive, dtype=torch.float32, device=device),
        }

        return total_loss, loss_dict


class HungarianLossV2(nn.Module):
    """
    Improved Hungarian loss with better handling of edge cases.

    This version:
    1. Uses focal loss for classification to handle class imbalance
    2. Supports smooth L1 loss for regression (more robust to outliers)
    3. Properly handles the case when all targets are background
    """

    def __init__(
        self,
        cfg,
    ):
        """
        Initialize the improved Hungarian loss.

        Args:
            cfg: Configuration node with:
                - CORONARY.NUM_PROPOSALS: Number of proposal tokens
                - CORONARY.NUM_BRANCHES: Number of branch classes (default: 4)
                - CORONARY.CLS_LOSS_WEIGHT: Weight for classification loss
                - CORONARY.REG_LOSS_WEIGHT: Weight for regression loss
                - HUNGARIAN: Hungarian matching parameters
                - LOSS_TYPE: 'multi_task' (default), 'focal', or 'smooth_l1'
        """
        super().__init__()

        self.cfg = cfg
        self.num_proposals = cfg.CORONARY.NUM_PROPOSALS
        self.num_branches = cfg.CORONARY.get('NUM_BRANCHES', 4)
        self.num_classes = self.num_branches + 1  # +1 for background
        self.cls_weight = cfg.CORONARY.CLS_LOSS_WEIGHT
        self.reg_weight = cfg.CORONARY.REG_LOSS_WEIGHT
        self.loss_type = cfg.CORONARY.get('LOSS_TYPE', 'hungarian')

        # Hungarian matcher
        hungarian_cfg = cfg.HUNGARIAN if hasattr(cfg, 'HUNGARIAN') else None
        cost_cls_weight = hungarian_cfg.COST_CLS_WEIGHT if hungarian_cfg else 1.0
        cost_reg_weight = hungarian_cfg.COST_REG_WEIGHT if hungarian_cfg else 1.0
        cost_reg_type = hungarian_cfg.COST_REG_TYPE if hungarian_cfg else 'mse'

        self.matcher = HungarianMatcher(
            cost_cls_weight=cost_cls_weight,
            cost_reg_weight=cost_reg_weight,
            cost_reg_type=cost_reg_type,
            num_classes=self.num_classes,
        )

        # Classification loss
        if self.loss_type == 'focal':
            self.cls_loss_fn = FocalLoss(
                alpha=cfg.CORONARY.get('FOCAL_ALPHA', 0.25),
                gamma=cfg.CORONARY.get('FOCAL_GAMMA', 2.0),
                reduction='none',
                num_classes=self.num_classes,
            )
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Regression loss
        if self.loss_type == 'smooth_l1':
            self.reg_loss_fn = SmoothL1Loss(
                beta=cfg.CORONARY.get('SMOOTH_L1_BETA', 1.0),
                reduction='none',
            )
        else:
            self.reg_loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Hungarian loss.

        Args:
            predictions: Model predictions containing:
                - 'cls_outputs': List of classification logits [batch, num_classes]
                - 'reg_outputs': List of regression outputs [batch, 1]
            targets: Ground truth targets containing:
                - 'cls_targets': Classification targets [batch, num_proposals]
                                (class indices: 0-3 for branches, 4 for background, padded with 0)
                - 'reg_targets': Regression targets [batch, num_proposals]
                - 'valid_mask': Validity mask [batch, num_proposals] (1 for real targets, 0 for padding)
        """
        cls_outputs = predictions['cls_outputs']  # List of [B, num_classes]
        reg_outputs = predictions['reg_outputs']  # List of [B, 1]

        cls_targets = targets['cls_targets']  # [B, N]
        reg_targets = targets['reg_targets']  # [B, N]
        valid_mask = targets.get('valid_mask', None)  # [B, N]

        batch_size = cls_targets.shape[0]
        device = cls_targets.device

        # Step 1: Hungarian matching (only for foreground targets: class 0-3)
        matches = self.matcher(
            cls_outputs=cls_outputs,
            reg_outputs=reg_outputs,
            cls_targets=cls_targets,
            reg_targets=reg_targets,
            valid_mask=valid_mask,
        )

        # Step 2: Compute losses
        cls_losses = []
        reg_losses = []
        num_positive = 0
        num_matched = 0

        for b in range(batch_size):
            pred_indices, target_indices = matches[b]

            # Get valid mask for this sample
            if valid_mask is not None:
                mask_b = valid_mask[b]  # [N]
            else:
                mask_b = torch.ones(self.num_proposals, device=device)

            # Losses for matched pairs (foreground targets only)
            for pred_idx, tgt_idx in zip(pred_indices, target_indices):
                # Only compute loss for valid foreground targets
                if mask_b[tgt_idx] > 0:
                    cls_pred = cls_outputs[pred_idx][b]  # [num_classes]
                    cls_tgt = cls_targets[b, tgt_idx].long()  # scalar (0-3 for foreground)

                    # Classification loss
                    cls_loss = self.cls_loss_fn(cls_pred.unsqueeze(0), cls_tgt.unsqueeze(0))
                    cls_losses.append(cls_loss)

                    # Regression loss for foreground targets only
                    reg_pred = reg_outputs[pred_idx][b, 0]
                    reg_tgt = reg_targets[b, tgt_idx]
                    reg_loss = self.reg_loss_fn(reg_pred, reg_tgt)
                    reg_losses.append(reg_loss)
                    num_positive += 1
                    num_matched += 1

            # Unmatched predictions should predict background
            matched_pred_set = set(pred_indices)
            for pred_idx in range(self.num_proposals):
                if pred_idx not in matched_pred_set:
                    cls_pred = cls_outputs[pred_idx][b]  # [num_classes]
                    bg_class = self.num_classes - 1  # Background class index
                    cls_tgt = torch.tensor(bg_class, dtype=torch.long, device=device)
                    cls_loss = self.cls_loss_fn(cls_pred.unsqueeze(0), cls_tgt.unsqueeze(0))
                    cls_losses.append(cls_loss)

        # Aggregate
        if len(cls_losses) > 0:
            cls_loss = torch.stack(cls_losses).mean()
        else:
            cls_loss = torch.zeros(1, device=device)[0]

        if len(reg_losses) > 0:
            reg_loss = torch.stack(reg_losses).mean()
        else:
            reg_loss = torch.zeros(1, device=device)[0]

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'num_positive': torch.tensor(num_positive, dtype=torch.float32, device=device),
            'num_matched': torch.tensor(num_matched, dtype=torch.float32, device=device),
        }

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification to handle class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'none',
        num_classes: int = 5,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, num_classes] logits
            targets: [batch] class indices
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=-1)

        # Get probability for target class
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        pt = torch.sum(probs * targets_one_hot, dim=-1)  # [batch]

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_t = torch.full_like(targets, self.alpha, dtype=torch.float32)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss) for robust regression.
    """

    def __init__(self, beta: float = 1.0, reduction: str = 'none'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


def build_hungarian_loss(cfg) -> nn.Module:
    """
    Build Hungarian loss function based on configuration.

    Args:
        cfg: Configuration node

    Returns:
        nn.Module: The loss module
    """
    loss_version = cfg.CORONARY.get('HUNGARIAN_LOSS_VERSION', 'v2')

    if loss_version == 'v1':
        return HungarianLoss(cfg)
    else:
        return HungarianLossV2(cfg)
