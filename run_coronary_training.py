#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冠状动脉多任务训练启动脚本

用于在远程服务器上启动冠状动脉 CT 视频分类和回归的联合训练

使用方法:
    # 本地训练
    python run_coronary_training.py

    # 远程训练
    python run_coronary_training.py --remote

    # 使用自定义配置
    python run_coronary_training.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml
"""

import os
import sys
import argparse

# 添加 SlowFast 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SlowFast'))

from utils.ssh_exec import SSHClient


def get_default_config():
    """获取默认配置文件路径"""
    return "configs/Coronary/MVITv2_B_32x3_multitask.yaml"


def run_local_training(cfg_file, extra_args=None):
    """在本地运行训练"""
    from slowfast.tools.run_net import launch_job
    from slowfast.config.defaults import assert_and_infer_cfg, get_cfg
    from slowfast.utils.misc import get_scratch_dir
    import torch

    # 加载配置
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)

    # 添加额外参数
    if extra_args:
        for i in range(0, len(extra_args), 2):
            key = extra_args[i]
            value = extra_args[i + 1]
            # 转换类型
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            cfg.merge_from_list([key, value])

    cfg = assert_and_infer_cfg(cfg)

    # 设置 GPU
    if not torch.cuda.is_available():
        print("警告：CUDA 不可用，将使用 CPU 训练")
        cfg.NUM_GPUS = 0

    # 启动训练
    from slowfast.tools.train_coronary_multitask import train
    train(cfg)


def run_remote_training(cfg_file, extra_args=None):
    """在远程服务器上运行训练"""

    # 构建命令
    cmd = f"cd /18018998051/CTA && "
    cmd += "source ~/.bashrc && "
    cmd += "conda activate mamba && "
    cmd += f"python tools/run_net.py --cfg {cfg_file}"

    if extra_args:
        cmd += " " + " ".join(extra_args)

    # 通过 SSH 执行
    with SSHClient() as ssh:
        ssh.exec_command(cmd, activate_conda=False)


def upload_code_to_remote():
    """上传代码到远程服务器"""
    print("上传代码到远程服务器...")

    files_to_upload = [
        # 配置文件
        ("E:/pycharm23/Projs/DcmDataset/SlowFast/configs/Coronary/MVITv2_B_32x3_multitask.yaml",
         "/18018998051/CTA/configs/Coronary/MVITv2_B_32x3_multitask.yaml"),

        # 模型文件
        ("E:/pycharm23/Projs/DcmDataset/SlowFast/slowfast/models/coronary_head.py",
         "/18018998051/CTA/slowfast/models/coronary_head.py"),
        ("E:/pycharm23/Projs/DcmDataset/SlowFast/slowfast/models/coronary_loss.py",
         "/18018998051/CTA/slowfast/models/coronary_loss.py"),

        # 训练文件
        ("E:/pycharm23/Projs/DcmDataset/SlowFast/slowfast/tools/train_coronary_multitask.py",
         "/18018998051/CTA/slowfast/tools/train_coronary_multitask.py"),

        # 数据集文件
        ("E:/pycharm23/Projs/DcmDataset/SlowFast/slowfast/datasets/coronary.py",
         "/18018998051/CTA/slowfast/datasets/coronary.py"),

        # 配置文件
        ("E:/pycharm23/Projs/DcmDataset/SlowFast/slowfast/config/defaults.py",
         "/18018998051/CTA/slowfast/config/defaults.py"),
    ]

    with SSHClient() as ssh:
        for local_path, remote_path in files_to_upload:
            if os.path.exists(local_path):
                ssh.upload_file(local_path, remote_path)
            else:
                print(f"警告：文件不存在 {local_path}")

    print("上传完成!")


def main():
    parser = argparse.ArgumentParser(description="冠状动脉多任务训练启动脚本")
    parser.add_argument("--remote", action="store_true", help="在远程服务器上运行")
    parser.add_argument("--upload", action="store_true", help="上传代码到远程服务器")
    parser.add_argument("--cfg", type=str, default=None, help="配置文件路径")
    parser.add_argument("extra_args", nargs="*", help="额外配置参数")

    args = parser.parse_args()

    # 确定配置文件
    cfg_file = args.cfg if args.cfg else get_default_config()

    if args.upload:
        upload_code_to_remote()

    if args.remote:
        print(f"在远程服务器上运行训练...")
        print(f"配置文件：{cfg_file}")
        run_remote_training(cfg_file, args.extra_args)
    else:
        print(f"在本地运行训练...")
        print(f"配置文件：{cfg_file}")
        run_local_training(cfg_file, args.extra_args)


if __name__ == "__main__":
    main()
