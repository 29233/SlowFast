#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import pprint

import numpy as np
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import torch
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from slowfast.models.coronary_loss import build_coronary_loss
from slowfast.models.hungarian_loss import build_hungarian_loss, HungarianMatcher
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)


def compute_ap_for_single_sample(cls_probs, reg_preds, cls_gts, reg_gts, thresholds, num_branches):
    """
    Compute mAP for a single sample.

    Args:
        cls_probs: [N, num_classes] tensor of class probabilities
        reg_preds: [N] tensor of regression predictions
        cls_gts: [N] tensor of class ground truth (0-3: foreground, 4: background)
        reg_gts: [N] tensor of regression ground truth
        thresholds: List of regression error thresholds for matching
        num_branches: Number of branch classes (default 4)

    Returns:
        ap_per_class: Dict of AP values for each class
        tp_total: Total TP count across all thresholds
        fp_total: Total FP count across all thresholds
        fn_total: Total FN count across all thresholds
        reg_errors: List of matched regression errors
    """
    num_proposals = cls_probs.shape[0]

    # Collect predictions and ground truths for each class
    all_preds_per_class = {c: [] for c in range(num_branches)}
    all_gts_per_class = {c: [] for c in range(num_branches)}

    # Collect ground truths (only foreground classes 0-3)
    for n in range(num_proposals):
        c = int(cls_gts[n].item())
        if c < num_branches:  # Foreground class
            all_gts_per_class[c].append((reg_gts[n].item(), n))

    # Collect predictions
    for p_idx in range(num_proposals):
        cls_prob = cls_probs[p_idx]  # [num_classes]
        cls_class = cls_prob.argmax().item()
        cls_conf = cls_prob.max().item()
        reg_pred_val = reg_preds[p_idx].item()

        if cls_class < num_branches:  # Only foreground predictions
            all_preds_per_class[cls_class].append((cls_conf, reg_pred_val, p_idx))

    # Sort predictions by confidence (descending) for each class
    for c in range(num_branches):
        all_preds_per_class[c].sort(key=lambda x: -x[0])

    # Compute AP for each class
    ap_per_class = {}
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    reg_errors = []

    for c in range(num_branches):
        all_preds_c = all_preds_per_class[c]
        all_gts_c = all_gts_per_class[c]
        num_gts = len(all_gts_c)

        if num_gts == 0:
            ap_per_class[c] = 0.0
            continue

        # Match predictions with ground truths using average threshold
        avg_thresh = sum(thresholds) / len(thresholds)
        matched_gt = set()

        for conf, reg_pred, p_idx in all_preds_c:
            # Find closest unmatched GT
            best_dist = float('inf')
            best_gt_idx = -1

            for gt_idx, (reg_gt, gt_n) in enumerate(all_gts_c):
                if gt_n not in matched_gt:
                    dist = abs(reg_pred - reg_gt)
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = gt_idx

            if best_dist <= avg_thresh and best_gt_idx >= 0:
                tp_total += 1
                matched_gt.add(all_gts_c[best_gt_idx][1])
                reg_errors.append(best_dist)
            else:
                fp_total += 1

        fn_total += num_gts - len(matched_gt)

        # Compute AP using 11-point interpolation
        if len(thresholds) > 0:
            ap_values = []
            for thresh in thresholds:
                matched_gt_thresh = set()
                tp_list = []
                fp_list = []

                for conf, reg_pred, p_idx in all_preds_c:
                    best_dist = float('inf')
                    best_gt_idx = -1

                    for gt_idx, (reg_gt, gt_n) in enumerate(all_gts_c):
                        if gt_n not in matched_gt_thresh:
                            dist = abs(reg_pred - reg_gt)
                            if dist < best_dist:
                                best_dist = dist
                                best_gt_idx = gt_idx

                    if best_dist <= thresh and best_gt_idx >= 0:
                        matched_gt_thresh.add(all_gts_c[best_gt_idx][1])
                        tp_list.append(1)
                        fp_list.append(0)
                    else:
                        tp_list.append(0)
                        fp_list.append(1)

                tp_cumsum = np.cumsum(tp_list)
                fp_cumsum = np.cumsum(fp_list)

                # 11-point interpolation
                recall_points = np.linspace(0, 1, 11)
                precision_values = []
                for r_target in recall_points:
                    valid_indices = tp_cumsum / num_gts >= r_target
                    if valid_indices.any():
                        precision_values.append((tp_cumsum[valid_indices] / (tp_cumsum[valid_indices] + fp_cumsum[valid_indices] + 1e-5)).max())
                    else:
                        precision_values.append(0.0)

                ap_values.append(np.mean(precision_values))

            ap_per_class[c] = np.mean(ap_values)
        else:
            ap_per_class[c] = 0.0

    return ap_per_class, tp_total, fp_total, fn_total, reg_errors


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    # 检查是否为 coronary_multitask 数据集
    is_coronary_multitask = cfg.TRAIN.DATASET == "coronary_multitask"

    # 检查是否使用匈牙利损失
    use_hungarian_loss = is_coronary_multitask and cfg.CORONARY.get('LOSS_TYPE', 'multi_task') == 'hungarian'

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)

    # 为 coronary_multitask 构建专用损失函数
    if is_coronary_multitask:
        if use_hungarian_loss:
            criterion = build_hungarian_loss(cfg)
        else:
            criterion = build_coronary_loss(cfg)
    else:
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            # 处理 metadata 中的字段，仅对张量类型调用 cuda()
            for key, val in meta.items():
                if isinstance(val, torch.Tensor):
                    # 张量类型直接转移到 GPU
                    meta[key] = val.cuda(non_blocking=True)
                elif isinstance(val, list):
                    # 列表类型：检查元素是否为张量
                    if len(val) > 0 and isinstance(val[0], torch.Tensor):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    # 非张量列表（如 video_id 字符串列表）保留在 CPU 上

        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()

            if cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                (
                    model,
                    preds,
                    partial_loss,
                    perform_backward,
                ) = contrastive_forward(
                    model, cfg, inputs, index, time, epoch_exact, scaler
                )
            elif cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
            elif cfg.MASK.ENABLE:
                preds, labels = model(inputs)
            else:
                preds = model(inputs)
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                labels = torch.zeros(
                    preds.size(0), dtype=labels.dtype, device=labels.device

                )

            # 为 coronary_multitask 计算专用损失
            if is_coronary_multitask:
                # 准备 targets 字典
                targets = {
                    'cls_targets': meta['cls_target'],
                    'reg_targets': meta['reg_target'],
                }
                if 'valid_mask' in meta:
                    targets['valid_mask'] = meta['valid_mask']
                # 计算多任务损失
                loss, loss_dict = criterion(preds, targets)
            elif cfg.MODEL.MODEL_NAME == "ContrastiveModel" and partial_loss:
                loss = partial_loss
            else:
                # Compute the loss.
                loss = loss_fun(preds, labels)

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())
        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        if update_param:
            scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        # 为 coronary_multitask 处理损失统计
        if is_coronary_multitask:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()
            # 从 loss_dict 中提取分类和回归损失
            cls_loss = loss_dict['cls_loss'].item() if 'cls_loss' in loss_dict else 0.0
            reg_loss = loss_dict['reg_loss'].item() if 'reg_loss' in loss_dict else 0.0

            # Update and log stats.
            train_meter.update_stats(
                top1_err=None, top5_err=None, loss=loss, lr=lr,
                grad_norm=None, mb_size=batch_size,
                cls_loss=cls_loss, reg_loss=reg_loss
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/cls_loss": cls_loss,
                        "Train/reg_loss": reg_loss,
                        "Train/lr": lr
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
        elif cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                loss, grad_norm = (
                    loss.item(),
                    grad_norm.item(),
                )
            elif cfg.MASK.ENABLE:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                    if loss_extra:
                        loss_extra = du.all_reduce(loss_extra)
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    0.0,
                    0.0,
                )
                if loss_extra:
                    loss_extra = [one_loss.item() for one_loss in loss_extra]
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm, top1_err, top5_err = du.all_reduce(
                        [loss.detach(), grad_norm, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                grad_norm,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                loss_extra,
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
    del inputs

    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    # 检查是否为 coronary_multitask 数据集
    is_coronary_multitask = cfg.TRAIN.DATASET == "coronary_multitask"

    # 检查是否使用匈牙利损失
    use_hungarian_loss = is_coronary_multitask and cfg.CORONARY.get('LOSS_TYPE', 'multi_task') == 'hungarian'

    # 为 coronary_multitask 构建专用损失函数
    if is_coronary_multitask:
        if use_hungarian_loss:
            criterion = build_hungarian_loss(cfg)
        else:
            criterion = build_coronary_loss(cfg)

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            # 处理 metadata 中的字段，仅对张量类型调用 cuda()
            for key, val in meta.items():
                if isinstance(val, torch.Tensor):
                    # 张量类型直接转移到 GPU
                    meta[key] = val.cuda(non_blocking=True)
                elif isinstance(val, list):
                    # 列表类型：检查元素是否为张量
                    if len(val) > 0 and isinstance(val[0], torch.Tensor):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    # 非张量列表（如 video_id 字符串列表）保留在 CPU 上
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )
        val_meter.data_toc()

        # 为 coronary_multitask 计算专用损失和指标
        if is_coronary_multitask:
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
                # Forward pass
                outputs = model(inputs)

                # 准备 targets 字典
                targets = {
                    'cls_targets': meta['cls_target'],
                    'reg_targets': meta['reg_target'],
                }
                if 'valid_mask' in meta:
                    targets['valid_mask'] = meta['valid_mask']

                # 计算多任务损失
                loss, loss_dict = criterion(outputs, targets)

            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]

            loss = loss.item()
            cls_loss = loss_dict['cls_loss'].item() if 'cls_loss' in loss_dict else 0.0
            reg_loss = loss_dict['reg_loss'].item() if 'reg_loss' in loss_dict else 0.0

            # =====================
            # 多标签分类 + 回归任务评价指标
            # =====================
            # 模型输出：
            #   - cls_outputs: [B, N, num_classes] logits (num_classes = 5: 4 branches + 1 background)
            #   - reg_outputs: [B, N] regression values
            #
            # 评估策略：
            #   - 多 token 模式 (USE_MULTI_TOKEN=True): 类似 DETR，使用 mAP 评估
            #   - 单 token 模式：直接逐位置比较
            # =====================

            num_classes = cfg.CORONARY.get('NUM_BRANCHES', 4) + 1  # 5 classes
            num_proposals = cfg.CORONARY.NUM_PROPOSALS
            num_branches = cfg.CORONARY.get('NUM_BRANCHES', 4)

            # 获取预测结果（新的张量格式）
            cls_outputs = outputs['cls_outputs']  # [B, N, num_classes]
            reg_outputs = outputs['reg_outputs']  # [B, N]

            # 获取真实标签
            cls_targets = meta['cls_target']  # [B, N] (0-3: foreground, 4: background)
            reg_targets = meta['reg_target']  # [B, N]
            valid_mask = meta.get('valid_mask', torch.ones_like(cls_targets).float())  # [B, N]

            if cfg.CORONARY.USE_MULTI_TOKEN and cls_outputs.shape[1] == num_proposals:
                # =====================
                # 多 token 模式：mAP 评估（类似 DETR）
                # =====================
                # 逐样本计算 mAP，然后汇总
                # 匹配规则：类别一致 + 回归误差小于阈值

                # 误差阈值列表 (类似 COCO)
                reg_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

                # 逐样本处理并汇总结果
                total_map = 0.0
                total_tp = 0.0
                total_fp = 0.0
                total_fn = 0.0
                all_reg_errors = []
                per_class_correct = [0.0] * num_branches
                per_class_count = [0.0] * num_branches

                for b in range(batch_size):
                    mask_b = valid_mask[b]
                    cls_gts_b = cls_targets[b]  # [N]
                    reg_gts_b = reg_targets[b]  # [N]
                    cls_probs_b = F.softmax(cls_outputs[b], dim=-1)  # [N, num_classes]
                    reg_preds_b = reg_outputs[b]  # [N]

                    # 计算该样本的 mAP（函数内部只处理 foreground 类别）
                    ap_per_class, tp, fp, fn, reg_errors = compute_ap_for_single_sample(
                        cls_probs_b, reg_preds_b, cls_gts_b, reg_gts_b,
                        reg_thresholds, num_branches,
                    )

                    # 汇总结果
                    sample_map = np.mean([ap_per_class[c] for c in range(num_branches)])
                    total_map += sample_map
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    all_reg_errors.extend(reg_errors)

                    # 统计每个类别的真实目标数量
                    for c in range(num_branches):
                        count = ((cls_gts_b == c).float() * mask_b).sum().item()
                        per_class_count[c] += count
                        per_class_correct[c] += ap_per_class[c] * count

                # 计算平均 mAP
                map_value = total_map / batch_size

                # 精确率、召回率、F1
                precision_overall = total_tp / (total_tp + total_fp + 1e-5)
                recall_overall = total_tp / (total_tp + total_fn + 1e-5)
                f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall + 1e-5)

                # 分类准确率（使用匹配结果）
                total_valid = (valid_mask > 0).sum().item()
                cls_accuracy = total_tp / max(total_valid, 1)

                # 前景分类准确率
                total_fg = sum(per_class_count)
                cls_accuracy_fg = total_tp / max(total_fg, 1)

                # 回归指标
                if len(all_reg_errors) > 0:
                    reg_mae = sum(all_reg_errors) / len(all_reg_errors)
                    reg_mse = sum(e ** 2 for e in all_reg_errors) / len(all_reg_errors)
                    reg_rmse = math.sqrt(reg_mse)
                else:
                    reg_mae = 0.0
                    reg_mse = 0.0
                    reg_rmse = 0.0

                # 计算每个类别的平均准确率
                per_class_correct = [per_class_correct[c] / max(per_class_count[c], 1) for c in range(num_branches)]
                per_class_count = per_class_count  # 已经是列表

            else:
                # =====================
                # 单 token 模式：直接逐位置比较
                # =====================
                # cls_outputs 已经是 [B, 1, num_classes] 格式
                cls_logits = cls_outputs  # [B, 1, num_classes]
                reg_pred = reg_outputs  # [B, 1]

                # 分类预测
                cls_probs = F.softmax(cls_logits, dim=-1)
                cls_pred_classes = cls_probs.argmax(dim=-1)

                # 分类准确率
                cls_correct = (cls_pred_classes == cls_targets).float() * valid_mask
                cls_accuracy = cls_correct.sum() / (valid_mask.sum() + 1e-5)

                # 前景分类准确率
                foreground_mask = (cls_targets < num_classes - 1).float() * valid_mask
                cls_correct_fg = (cls_pred_classes == cls_targets).float() * foreground_mask
                cls_accuracy_fg = cls_correct_fg.sum() / (foreground_mask.sum() + 1e-5)

                # 每个分支类别的准确率
                per_class_correct = []
                per_class_count = []
                for c in range(num_branches):
                    class_mask = (cls_targets == c).float() * valid_mask
                    class_correct = (cls_pred_classes == cls_targets).float() * class_mask
                    per_class_correct.append(class_correct.sum().item())
                    per_class_count.append(class_mask.sum().item())

                # 前景召回率
                pred_foreground_mask = (cls_pred_classes < num_classes - 1).float()
                true_positive = (pred_foreground_mask * foreground_mask).sum()
                pred_positive = pred_foreground_mask.sum()
                gt_positive = foreground_mask.sum()

                precision_overall = true_positive / (pred_positive + 1e-5)
                recall_overall = true_positive / (gt_positive + 1e-5)
                f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall + 1e-5)

                # 回归指标
                if foreground_mask.sum() > 0:
                    reg_mae = (torch.abs(reg_pred - reg_targets) * foreground_mask).sum() / foreground_mask.sum()
                    reg_mse = ((reg_pred - reg_targets) ** 2 * foreground_mask).sum() / foreground_mask.sum()
                    reg_rmse = torch.sqrt(reg_mse)
                else:
                    reg_mae = torch.tensor(0.0)
                    reg_mse = torch.tensor(0.0)
                    reg_rmse = torch.tensor(0.0)

                # mAP 在单 token 模式下用类别平均准确率代替
                map_value = cls_accuracy.item()
                ap_per_class = {c: per_class_correct[c] / (per_class_count[c] + 1e-5) for c in range(num_branches)}

            # =====================
            # 记录指标
            # =====================
            val_meter.update_stats(
                top1_err=None, top5_err=None, mb_size=batch_size,
                loss=loss, cls_loss=cls_loss, reg_loss=reg_loss,
                cls_accuracy=cls_accuracy if isinstance(cls_accuracy, float) else cls_accuracy.item(),
                cls_accuracy_fg=cls_accuracy_fg if isinstance(cls_accuracy_fg, float) else cls_accuracy_fg.item(),
                precision=precision_overall,
                recall=recall_overall,
                f1_score=f1_overall,
                reg_mae=reg_mae if isinstance(reg_mae, float) else reg_mae.item(),
                reg_mse=reg_mse if isinstance(reg_mse, float) else reg_mse.item(),
                reg_rmse=reg_rmse if isinstance(reg_rmse, float) else reg_rmse.item(),
                per_class_correct=per_class_correct,
                per_class_count=per_class_count,
            )
        elif cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if not cfg.CONTRASTIVE.KNN_ON:
                    return
                train_labels = (
                    model.module.train_labels
                    if hasattr(model, "module")
                    else model.train_labels
                )
                yd, yi = model(inputs, index, time)
                K = yi.shape[1]
                C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                probs = torch.mul(
                    retrieval_one_hot.view(batch_size, -1, C),
                    yd_transform.view(batch_size, -1, 1),
                )
                preds = torch.sum(probs, 1)
            else:
                preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                if cfg.DATA.IN22k_VAL_IN1K != "":
                    preds = preds[:, :1000]
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars({"Val/mAP": val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [label.clone().detach() for label in val_meter.all_labels]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    flops, params = 0.0, 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            train_loader = loader.construct_loader(cfg, "train")
            loader.shuffle_dataset(train_loader, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        # train_epoch(
        #     train_loader,
        #     model,
        #     optimizer,
        #     scaler,
        #     train_meter,
        #     cur_epoch,
        #     cfg,
        #     writer,
        # )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time() / len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time() / len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            and not cfg.MASK.ENABLE
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
                writer,
            )
    if (
        start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE
    ):  # final checkpoint load
        eval_epoch(val_loader, model, val_meter, start_epoch, cfg, train_loader, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            (
                epoch_timer.median_epoch_time() / 60.0
                if len(epoch_timer.epoch_times)
                else 0.0
            ),
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))

    return result_string
