# 冠状动脉 CT 血管造影多任务学习项目

本项目基于 SlowFast 视频理解框架，实现了冠状动脉 CT 血管造影 (CCTA) 的多任务深度学习模型，可同时进行斑块检测（分类）和狭窄程度评估（回归）。

## 项目概述

### 核心功能

- **多任务学习**: 联合训练斑块分类和狭窄程度回归任务
- **多目标检测**: 支持单样本多斑块目标（类似检测任务范式）
- **MViTv2 Backbone**: 采用 Multiscale Vision Transformer 作为主干网络
- **混合精度训练**: 支持 AMP 加速训练
- **TensorBoard 可视化**: 完整的训练指标和预测可视化

### 技术特点

1. **数据加载**: 按患者 ID 聚合同一血管的多个斑块目标
2. **Valid Mask**: 自动区分真实目标和 padding
3. **分支编码**: 使用 one-hot 编码标识血管分支类型（LAD, RCA, LCX, LM）
4. **多 Token 输出**: 可选的多 proposal token 输出模式

## 环境配置

### 系统要求

- Python >= 3.8
- CUDA >= 11.0
- GCC >= 4.9

### 依赖安装

```bash
# 1. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision torchaudio

# 2. 安装基础依赖
pip install simplejson psutil opencv-python tensorboard moviepy

# 3. 安装 fvcore 和 iopath
pip install 'git+https://github.com/facebookresearch/fvcore.git'
pip install -U iopath

# 4. 安装 PyAV（视频解码）
conda install av -c conda-forge

# 5. 安装 PyTorchVideo
pip install pytorchvideo

# 6. 安装 Detectron2
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
cd detectron2_repo
pip install -e .
cd ..

# 7. 安装 SlowFast
cd SlowFast
python setup.py build develop
cd ..
```

### 验证安装

```bash
python -c "import slowfast; print('SlowFast 安装成功!')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pytorchvideo; print('PyTorchVideo 安装成功!')"
```

## 项目结构

```
SlowFast/
├── configs/Coronary/
│   ├── MVITv2_B_32x3.yaml              # 单任务配置文件
│   └── MVITv2_B_32x3_multitask.yaml    # 多任务配置文件
├── slowfast/
│   ├── datasets/
│   │   └── coronary.py                 # 冠脉数据加载器
│   ├── models/
│   │   ├── video_model_builder.py      # 模型构建器（含冠脉头）
│   │   ├── coronary_head.py            # 多任务预测头
│   │   └── coronary_loss.py            # 多任务损失函数
│   ├── utils/
│   │   └── meters.py                   # 训练指标统计
│   └── tools/
│       ├── train_net.py                # 训练脚本
│       ├── test_net.py                 # 测试脚本
│       └── run_net.py                  # 运行入口
└── README_CORONARY.md                  # 项目说明文档
```

## 数据准备

### 数据格式要求

1. **DICOM 序列文件**: 按患者 ID 组织的文件夹结构
```
/18018998051/data/Central/CTA/
├── 870328/
│   ├── series1.dcm
│   └── series2.dcm
├── 828085/
│   └── ...
└── ...
```

2. **metadata 文件**: `metadataV1.csv` 包含以下字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ID | str | 患者唯一标识 |
| Plaque | float | 斑块百分比值 (0-100) |
| Branch | str | 血管分支类型 (LAD/RCA/LCX/LM) |
| Split | str | 数据划分 (Train/Val/Test) |

### 数据加载逻辑

相同 ID 的多个斑块会自动聚合为一个样本，每个样本包含：
- 视频数据：[C, T, H, W] 格式
- 分类标签：每个斑块的置信度目标
- 回归标签：每个斑块的狭窄程度
- 分支编码：one-hot 向量表示血管类型
- Valid Mask: 区分真实目标和 padding

## 训练配置

### 核心配置项

在 `configs/Coronary/MVITv2_B_32x3_multitask.yaml` 中修改：

```yaml
# 数据配置
DATA:
  NUM_FRAMES: 16              # 输入帧数
  SAMPLING_RATE: 1            # 采样率
  BATCH_SIZE: 4               # 批次大小
  PATH_TO_DATA_DIR: /path/to/data
  PATH_PREFIX: /path/to/dcm

# 模型配置
MODEL:
  MODEL_NAME: MViT
  NUM_CLASSES: 1
  LOSS_FUNC: multi_mse

# 冠脉多任务配置
CORONARY:
  USE_MULTI_TOKEN: True          # 是否使用多 token 输出
  NUM_PROPOSALS: 10              # 输出 token 数量
  CONFIDENCE_THRESHOLD: 0.5      # 测试时置信度阈值
  CLS_HIDDEN_DIM: 256            # 分类头隐藏层维度
  REG_HIDDEN_DIM: 256            # 回归头隐藏层维度
  CLS_LOSS_WEIGHT: 1.0           # 分类损失权重
  REG_LOSS_WEIGHT: 1.0           # 回归损失权重
  LOSS_TYPE: 'multi_task'        # 损失类型：multi_task/focal/smooth_l1
  PLAQUE_NORM_FACTOR: 100.0      # 斑块归一化因子

# 优化器配置
SOLVER:
  BASE_LR: 0.00005
  LR_POLICY: cosine
  MAX_EPOCH: 100
  WARMUP_EPOCHS: 10.0
  OPTIMIZING_METHOD: adamw
  WEIGHT_DECAY: 0.05

# 训练设置
TRAIN:
  ENABLE: True
  EVAL_PERIOD: 5              # 验证周期（epoch）
  CHECKPOINT_PERIOD: 5        # 检查点保存周期
```

## 训练命令

### 本地训练

```bash
# 进入项目目录
cd E:\pycharm23\Projs\DcmDataset\SlowFast

# 使用多 GPU 训练
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml \
    NUM_GPUS 2 \
    TRAIN.BATCH_SIZE 8 \
    DATA.PATH_TO_DATA_DIR /path/to/data

# 单 GPU 训练
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml \
    NUM_GPUS 1 \
    TRAIN.BATCH_SIZE 4
```

### 远程服务器训练

```bash
# 通过 SSH 在远程服务器执行
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml \
    DATA.PATH_PREFIX /18018998051/data/Central/CTA \
    OUTPUT_DIR ./outputs/coronary_multitask
```

### 断点续训练

```bash
# 自动恢复最近的检查点
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml \
    TRAIN.AUTO_RESUME True \
    CHECKPOINT_FILE_PATH ./outputs/checkpoint_epoch_00050.pyth
```

## 测试和验证

### 多视角测试

```bash
# 默认测试
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml \
    TRAIN.ENABLE False \
    TEST.ENABLE True

# 多视角集成测试
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml \
    TEST.NUM_ENSEMBLE_VIEWS 5
```

### 测试输出

测试完成后会输出：
- 分类准确率
- 回归 MAE（平均绝对误差）
- 回归 MSE（均方误差）
- 各血管分支的性能统计

## 模型输出

### 预测格式

模型输出为字典格式：

```python
outputs = {
    'cls_outputs': [tensor1, tensor2, ...],  # 分类输出列表 [batch, 1]
    'reg_outputs': [tensor1, tensor2, ...],  # 回归输出列表 [batch, 1]
    'all_tokens': tensor                      # 所有 proposal tokens
}
```

### 后处理

```python
# 多 token 模式：平均所有 proposal 的预测
if USE_MULTI_TOKEN:
    cls_pred = torch.stack(outputs['cls_outputs']).mean(0)
    reg_pred = torch.stack(outputs['reg_outputs']).mean(0)
else:
    cls_pred = outputs['cls_outputs'][0]
    reg_pred = outputs['reg_outputs'][0]

# 应用置信度阈值过滤
predictions = []
for i, conf in enumerate(cls_pred):
    if conf >= CONFIDENCE_THRESHOLD:
        predictions.append({
            'confidence': conf.item(),
            'plaque_percentage': reg_pred[i].item(),
            'branch': branch_names[i]
        })
```

## 训练指标说明

### 训练阶段指标

- **total_loss**: 总损失（加权和）
- **cls_loss**: 分类损失（BCE）
- **reg_loss**: 回归损失（MSE）
- **lr**: 当前学习率
- **grad_norm**: 梯度范数

### 验证阶段指标

- **top1_err**: 分类错误率
- **cls_accuracy**: 分类准确率（使用置信度阈值）
- **reg_mae**: 回归平均绝对误差
- **reg_mse**: 回归均方误差
- **loss**: 验证集总损失

### TensorBoard 可视化

启动训练后，使用以下命令查看可视化：

```bash
tensorboard --logdir ./outputs/coronary_multitask/runs
```

可视化内容包括：
- 损失曲线（total_loss, cls_loss, reg_loss）
- 指标曲线（accuracy, mae, mse）
- 学习率变化
- 梯度分布

## 常见问题

### Q1: 如何调整输出 token 数量？

修改配置文件中的 `CORONARY.NUM_PROPOSALS` 参数：
```yaml
CORONARY:
  NUM_PROPOSALS: 20  # 增加到 20 个输出
```

### Q2: 如何调整置信度阈值？

修改配置文件中的 `CORONARY.CONFIDENCE_THRESHOLD` 参数：
```yaml
CORONARY:
  CONFIDENCE_THRESHOLD: 0.3  # 降低阈值，检测更多斑块
```

### Q3: 如何修改损失权重？

调整分类和回归损失的权重比例：
```yaml
CORONARY:
  CLS_LOSS_WEIGHT: 2.0  # 增加分类损失权重
  REG_LOSS_WEIGHT: 1.0  # 回归损失权重不变
```

### Q4: 如何切换到单 token 模式？

设置 `USE_MULTI_TOKEN` 为 False：
```yaml
CORONARY:
  USE_MULTI_TOKEN: False  # 单 token 模式
```

### Q5: 如何处理类别不平衡问题？

可以使用 Focal Loss：
```yaml
CORONARY:
  LOSS_TYPE: 'focal'
  FOCAL_ALPHA: 0.25
  FOCAL_GAMMA: 2.0
```

## 故障排除

### CUDA 内存不足

减少批次大小或帧数：
```yaml
DATA:
  BATCH_SIZE: 2  # 减小批次
  NUM_FRAMES: 8  # 减少帧数
```

### 数据加载慢

增加数据加载 worker 数量：
```yaml
DATA_LOADER:
  NUM_WORKERS: 4
  PIN_MEMORY: True
```

### 训练不收敛

检查以下配置：
- 学习率是否过高（建议从 0.00005 开始）
- Warmup epochs 是否足够（建议 10 epochs）
- 数据归一化是否正确（PLAQUE_NORM_FACTOR）

## 参考资料

- [SlowFast 官方仓库](https://github.com/facebookresearch/SlowFast)
- [MViT 论文](https://arxiv.org/abs/2104.11227)
- [MViTv2 论文](https://arxiv.org/abs/2112.01526)
- [PyTorchVideo 文档](https://pytorchvideo.org/)

## 许可证

本项目基于 SlowFast（Apache 2.0 许可证），冠脉多任务扩展代码遵循相同许可证。
