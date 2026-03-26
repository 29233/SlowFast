#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Coronary Multi-Task Head for Classification and Regression.

This module implements a multi-token head that:
1. Outputs multiple proposal tokens from the class token
2. Each token has a classification head (multi-class: background + 4 branches) and regression head (0-1 plaque)
3. Supports configurable number of output tokens and thresholds

For multi-class classification:
- num_classes = num_branches + 1 (4 foreground classes: LAD, RCA, LCX, LM + 1 background class)
- Classification head outputs logits for each class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# Default branch classes for coronary CT
DEFAULT_BRANCHES = ['LAD', 'RCA', 'LCX', 'LM']


class CoronaryMultiTaskHead(nn.Module):
    """
    Multi-task head for coronary CT analysis with multi-class classification.

    Outputs multiple proposal tokens, each with:
    - Classification head: outputs logits for num_classes (num_branches + 1 background)
    - Regression head: outputs normalized plaque percentage (0-1)

    Attributes:
        num_proposals (int): Number of proposal tokens to output
        num_classes (int): Number of classification classes (num_branches + 1 for background)
        confidence_threshold (float): Threshold for filtering proposals at test time
        hidden_dim (int): Dimension of hidden layers
    """

    def __init__(
        self,
        dim_in: int,
        cfg,
    ):
        """
        Initialize the multi-task head.

        Args:
            dim_in (int): Input feature dimension (from the class token)
            cfg: Configuration node with the following required fields:
                - CORONARY.NUM_PROPOSALS: Number of proposal tokens
                - CORONARY.CONFIDENCE_THRESHOLD: Confidence threshold for filtering
                - CORONARY.CLS_HIDDEN_DIM: Hidden dimension for classification head
                - CORONARY.REG_HIDDEN_DIM: Hidden dimension for regression head
                - CORONARY.USE_MULTI_TOKEN: Whether to use multi-token output
                - CORONARY.NUM_BRANCHES: Number of branch classes (default: 4)
        """
        super(CoronaryMultiTaskHead, self).__init__()

        self.cfg = cfg
        self.num_proposals = cfg.CORONARY.NUM_PROPOSALS
        self.confidence_threshold = cfg.CORONARY.CONFIDENCE_THRESHOLD
        self.use_multi_token = cfg.CORONARY.USE_MULTI_TOKEN

        # Get number of branch classes (default 4: LAD, RCA, LCX, LM)
        # num_classes = num_branches + 1 (including background)
        self.num_branches = cfg.CORONARY.get('NUM_BRANCHES', 4)
        self.num_classes = self.num_branches + 1  # +1 for background class

        # Get hidden dimensions
        cls_hidden_dim = cfg.CORONARY.CLS_HIDDEN_DIM
        reg_hidden_dim = cfg.CORONARY.REG_HIDDEN_DIM

        if self.use_multi_token:
            # Multi-token mode: generate multiple proposal tokens
            # Token generation layer: dim_in -> (num_proposals x dim_in)
            self.token_generator = nn.Sequential(
                nn.Linear(dim_in, dim_in * 2),
                nn.GELU(),
                nn.LayerNorm(dim_in * 2),
                nn.Linear(dim_in * 2, dim_in * self.num_proposals),
                nn.GELU(),
            )

            # LayerNorm after token generation
            self.token_norm = nn.LayerNorm(dim_in)

            # Classification head for each proposal (multi-class: num_branches + 1)
            self.cls_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_in, cls_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.MVIT.DROPOUT_RATE if hasattr(cfg.MVIT, 'DROPOUT_RATE') else 0.1),
                    nn.Linear(cls_hidden_dim, self.num_classes)  # Output: num_classes logits
                ) for _ in range(self.num_proposals)
            ])

            # Regression head for each proposal
            self.reg_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_in, reg_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.MVIT.DROPOUT_RATE if hasattr(cfg.MVIT, 'DROPOUT_RATE') else 0.1),
                    nn.Linear(reg_hidden_dim, 1),
                    nn.Sigmoid()
                ) for _ in range(self.num_proposals)
            ])
        else:
            # Single token mode: use class token directly
            # Classification head (multi-class)
            self.cls_head = nn.Sequential(
                nn.Linear(dim_in, cls_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.MVIT.DROPOUT_RATE if hasattr(cfg.MVIT, 'DROPOUT_RATE') else 0.1),
                nn.Linear(cls_hidden_dim, self.num_classes)  # Output: num_classes logits
            )

            # Regression head
            self.reg_head = nn.Sequential(
                nn.Linear(dim_in, reg_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.MVIT.DROPOUT_RATE if hasattr(cfg.MVIT, 'DROPOUT_RATE') else 0.1),
                nn.Linear(reg_hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task head.

        Args:
            x (torch.Tensor): Input features.
                - Training: [batch_size, dim_in] (class token)
                - Testing: [batch_size, dim_in] (class token)
            return_all_tokens (bool): Whether to return all tokens (for debugging)

        Returns:
            Dict containing:
                - 'cls_outputs': List of classification logits [batch, num_classes] for each proposal
                - 'reg_outputs': List of regression outputs [batch, 1] for each proposal
                - 'all_tokens': All proposal tokens [batch, num_proposals, dim_in] (if use_multi_token)
        """
        if self.use_multi_token:
            # Generate multiple proposal tokens
            batch_size = x.shape[0]

            # Token generation: [B, D] -> [B, num_proposals * D] -> [B, num_proposals, D]
            tokens = self.token_generator(x)
            tokens = tokens.view(batch_size, self.num_proposals, -1)
            tokens = self.token_norm(tokens)  # [B, N, D]

            # Get outputs for each proposal
            cls_outputs = []
            reg_outputs = []

            for i in range(self.num_proposals):
                token_i = tokens[:, i, :]  # [B, D]
                cls_out = self.cls_head[i](token_i)  # [B, num_classes]
                reg_out = self.reg_head[i](token_i)  # [B, 1]
                cls_outputs.append(cls_out)
                reg_outputs.append(reg_out)

            result = {
                'cls_outputs': cls_outputs,  # List of [B, num_classes]
                'reg_outputs': reg_outputs,  # List of [B, 1]
                'all_tokens': tokens,  # [B, N, D]
            }
        else:
            # Single token mode
            cls_out = self.cls_head(x)  # [B, num_classes]
            reg_out = self.reg_head(x)  # [B, 1]

            result = {
                'cls_outputs': [cls_out],
                'reg_outputs': [reg_out],
            }

        return result


class CoronarySimpleHead(nn.Module):
    """
    Simple single-token head for coronary CT analysis with multi-class classification.

    Outputs:
    - Classification: logits for num_classes (num_branches + 1 background)
    - Regression: normalized plaque (0-1)
    """

    def __init__(
        self,
        dim_in: int,
        cfg,
    ):
        """
        Initialize the simple head.

        Args:
            dim_in (int): Input feature dimension
            cfg: Configuration node
        """
        super(CoronarySimpleHead, self).__init__()

        self.cfg = cfg
        self.confidence_threshold = cfg.CORONARY.CONFIDENCE_THRESHOLD

        # Get number of branch classes (default 4: LAD, RCA, LCX, LM)
        self.num_branches = cfg.CORONARY.get('NUM_BRANCHES', 4)
        self.num_classes = self.num_branches + 1  # +1 for background class

        # Classification head (multi-class)
        cls_hidden_dim = cfg.CORONARY.CLS_HIDDEN_DIM
        self.cls_fc = nn.Sequential(
            nn.Linear(dim_in, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.MVIT.DROPOUT_RATE if hasattr(cfg.MVIT, 'DROPOUT_RATE') else 0.1),
            nn.Linear(cls_hidden_dim, self.num_classes)  # Output: num_classes logits
        )

        # Regression head
        reg_hidden_dim = cfg.CORONARY.REG_HIDDEN_DIM
        self.reg_fc = nn.Sequential(
            nn.Linear(dim_in, reg_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.MVIT.DROPOUT_RATE if hasattr(cfg.MVIT, 'DROPOUT_RATE') else 0.1),
            nn.Linear(reg_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features [batch, dim_in]

        Returns:
            Dict containing:
                - 'cls_logits': Classification logits [batch, num_classes]
                - 'reg_pred': Regression predictions [batch, 1]
        """
        cls_logits = self.cls_fc(x)  # [B, num_classes]
        reg_pred = self.reg_fc(x)  # [B, 1]

        return {
            'cls_logits': cls_logits,
            'reg_pred': reg_pred,
        }


def build_coronary_head(dim_in: int, cfg):
    """
    Build coronary head based on configuration.

    Args:
        dim_in (int): Input feature dimension
        cfg: Configuration node

    Returns:
        nn.Module: The head module
    """
    if cfg.CORONARY.USE_MULTI_TOKEN:
        return CoronaryMultiTaskHead(dim_in, cfg)
    else:
        return CoronarySimpleHead(dim_in, cfg)
