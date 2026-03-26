#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a coronary multi-task model for classification and regression."""

import pprint

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.models.coronary_loss import build_coronary_loss
from slowfast.utils.meters import EpochTimer, TrainMeter, ValMeter

logger = logging.get_logger(__name__)


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
    Perform the coronary multi-task training for one epoch.

    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's parameters.
        scaler (GradScaler): gradient scaler for mixed precision training.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs.
        writer (TensorboardWriter, optional): TensorboardWriter object to write Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    # Build multi-task loss
    criterion = build_coronary_loss(cfg)

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)

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

            # Move targets to GPU
            cls_targets = meta['cls_target'].cuda(non_blocking=True)
            reg_targets = meta['reg_target'].cuda(non_blocking=True)
            valid_mask = meta.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )

        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Returns dict with cls_outputs and reg_outputs

            # Prepare targets dictionary
            targets = {
                'cls_targets': cls_targets,
                'reg_targets': reg_targets,
            }
            if valid_mask is not None:
                targets['valid_mask'] = valid_mask

            # Compute multi-task loss
            loss, loss_dict = criterion(outputs, targets)

        # Check for NaN loss
        misc.check_nan_losses(loss)

        # Backward pass
        scaler.scale(loss).backward()

        # Unscales the gradients
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

        # Update parameters
        scaler.step(optimizer)
        scaler.update()

        # Gather stats across GPUs
        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]

        loss = loss.item()
        cls_loss = loss_dict['cls_loss'].item() if 'cls_loss' in loss_dict else 0.0
        reg_loss = loss_dict['reg_loss'].item() if 'reg_loss' in loss_dict else 0.0

        # Update and log stats.
        train_meter.update_stats(
            None,  # No top1_acc for regression
            None,  # No top5_acc
            batch_size,
            loss,
            lr,
            cls_loss=cls_loss,
            reg_loss=reg_loss,
        )

        # Write to tensorboard if available
        if writer is not None:
            global_step = data_size * cur_epoch + cur_iter
            writer.add_scalars(
                {
                    "Train/total_loss": loss,
                    "Train/cls_loss": cls_loss,
                    "Train/reg_loss": reg_loss,
                    "Train/lr": lr,
                },
                global_step=global_step,
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    train_meter.epoch_iter_finished()
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader,
    model,
    val_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform coronary multi-task evaluation for one epoch.

    Args:
        val_loader (loader): video validation loader.
        model (model): the video model to evaluate.
        val_meter (ValMeter): validation meters to log the evaluation performance.
        cur_epoch (int): current epoch of evaluation.
        cfg (CfgNode): configs.
        writer (TensorboardWriter, optional): TensorboardWriter object to write Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    val_meter.iter_tic()

    # Build multi-task loss
    criterion = build_coronary_loss(cfg)

    # For collecting predictions and ground truths
    all_cls_preds = []
    all_reg_preds = []
    all_cls_targets = []
    all_reg_targets = []

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
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

            # Move targets to GPU
            cls_targets = meta['cls_target'].cuda(non_blocking=True)
            reg_targets = meta['reg_target'].cuda(non_blocking=True)
            valid_mask = meta.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )

        val_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Forward pass
            outputs = model(inputs)

            # Prepare targets dictionary
            targets = {
                'cls_targets': cls_targets,
                'reg_targets': reg_targets,
            }
            if valid_mask is not None:
                targets['valid_mask'] = valid_mask

            # Compute multi-task loss
            loss, loss_dict = criterion(outputs, targets)

        # Gather stats across GPUs
        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]

        loss = loss.item()
        cls_loss = loss_dict['cls_loss'].item() if 'cls_loss' in loss_dict else 0.0
        reg_loss = loss_dict['reg_loss'].item() if 'reg_loss' in loss_dict else 0.0

        # Collect predictions and targets for metrics
        if cfg.CORONARY.USE_MULTI_TOKEN:
            # For multi-token, average the predictions across proposals
            cls_pred = torch.stack(outputs['cls_outputs']).mean(0).squeeze(-1)  # [batch]
            reg_pred = torch.stack(outputs['reg_outputs']).mean(0).squeeze(-1)  # [batch]
        else:
            cls_pred = outputs['cls_outputs'][0].squeeze(-1)  # [batch]
            reg_pred = outputs['reg_outputs'][0].squeeze(-1)  # [batch]

        all_cls_preds.append(cls_pred.cpu())
        all_reg_preds.append(reg_pred.cpu())

        # For targets, compute the mean across valid proposals
        if valid_mask is not None:
            # Use valid_mask to compute weighted average of targets
            cls_target_mean = (cls_targets * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-5)
            reg_target_mean = (reg_targets * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-5)
        else:
            cls_target_mean = cls_targets.mean(dim=1)
            reg_target_mean = reg_targets.mean(dim=1)

        all_cls_targets.append(cls_target_mean.cpu())
        all_reg_targets.append(reg_target_mean.cpu())

        val_meter.update_stats(
            None,
            None,
            batch_size,
            loss,
            cls_loss=cls_loss,
            reg_loss=reg_loss,
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Compute metrics
    all_cls_preds = torch.cat(all_cls_preds, dim=0)
    all_reg_preds = torch.cat(all_reg_preds, dim=0)
    all_cls_targets = torch.cat(all_cls_targets, dim=0)
    all_reg_targets = torch.cat(all_reg_targets, dim=0)

    # Classification metrics (using threshold)
    threshold = cfg.CORONARY.CONFIDENCE_THRESHOLD
    cls_preds_binary = (all_cls_preds >= threshold).float()
    cls_correct = (cls_preds_binary == all_cls_targets).float().mean().item()

    # Regression metrics (MAE and MSE)
    reg_mae = torch.abs(all_reg_preds - all_reg_targets).mean().item()
    reg_mse = ((all_reg_preds - all_reg_targets) ** 2).mean().item()

    val_meter.epoch_iter_finished()
    val_meter.reset_metrics({
        'cls_accuracy': cls_correct,
        'reg_mae': reg_mae,
        'reg_mse': reg_mse,
    })

    # Log epoch stats
    val_meter.log_epoch_stats(cur_epoch)

    # Write to tensorboard
    if writer is not None:
        writer.add_scalars(
            {
                "Val/total_loss": loss,
                "Val/cls_accuracy": cls_correct,
                "Val/reg_mae": reg_mae,
                "Val/reg_mse": reg_mse,
            },
            global_step=cur_epoch,
        )


def train(cfg):
    """
    Train a coronary multi-task model.

    Args:
        cfg (CfgNode): configs.
    """
    # Set up environment.
    du.init_process_group()

    # Set random seed.
    misc.set_random_seed(cfg.RNG_SEED)

    # Set up model.
    model = build_model(cfg)

    # Set up optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Set up GradScaler for mixed precision.
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Resume training from checkpoint.
    start_epoch = 0
    if cu.has_checkpoint(cfg):
        checkpoint = cu.load_checkpoint(cfg)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Move model to GPU.
    if cfg.NUM_GPUS:
        model = du.nn.DataParallel(model, device_ids=list(range(cfg.NUM_GPUS)))
        model.cuda()

    # Set up training loader.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Set up training meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Set up Tensorboard writer.
    writer = tb.TensorboardWriter(cfg) if cfg.TENSORBOARD.ENABLE else None

    # Training loop.
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )

        # Evaluate on validation set.
        if (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                writer,
            )

        # Save checkpoint.
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            cu.save_checkpoint(cfg, model, optimizer, cur_epoch)

    # Final evaluation.
    eval_epoch(
        val_loader,
        model,
        val_meter,
        cfg.SOLVER.MAX_EPOCH,
        cfg,
        writer,
    )

    # Save final checkpoint.
    cu.save_checkpoint(cfg, model, optimizer, cfg.SOLVER.MAX_EPOCH)

    logger.info("Training completed!")
