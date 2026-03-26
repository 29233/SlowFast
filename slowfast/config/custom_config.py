#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs and default values.

    # Coronary Multi-Task Learning Configuration
    # Note: Main config is defined in slowfast/config/defaults.py
    # This is kept for backward compatibility
    _C.CORONARY = CfgNode()
    _C.CORONARY.USE_MULTI_TOKEN = False
    _C.CORONARY.NUM_PROPOSALS = 10
    _C.CORONARY.CONFIDENCE_THRESHOLD = 0.5
    _C.CORONARY.CLS_HIDDEN_DIM = 256
    _C.CORONARY.REG_HIDDEN_DIM = 256
    _C.CORONARY.CLS_LOSS_WEIGHT = 1.0
    _C.CORONARY.REG_LOSS_WEIGHT = 1.0
    _C.CORONARY.LOSS_TYPE = 'multi_task'
    _C.CORONARY.FOCAL_ALPHA = 0.25
    _C.CORONARY.FOCAL_GAMMA = 2.0
    _C.CORONARY.SMOOTH_L1_BETA = 1.0
    _C.CORONARY.LABEL_SMOOTH = 0.0
    _C.CORONARY.PLAQUE_NORM_FACTOR = 100.0
    _C.CORONARY.HUNGARIAN_LOSS_VERSION = 'v2'  # 'v1' or 'v2'
    _C.CORONARY.NUM_BRANCHES = 4  # Number of branch classes: LAD, RCA, LCX, LM

    # Hungarian Matching Configuration for Set-based Prediction
    _C.HUNGARIAN = CfgNode()
    _C.HUNGARIAN.COST_CLS_WEIGHT = 1.0  # Classification cost weight in matching
    _C.HUNGARIAN.COST_REG_WEIGHT = 1.0  # Regression cost weight in matching
    _C.HUNGARIAN.COST_REG_TYPE = 'mse'  # Regression cost type: 'mse' or 'l1'
