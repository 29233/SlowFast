# 冠脉 CT 数据集 SlowFast 训练快速入门指南

## 1. 文件结构

已创建的文件：

```
SlowFast/
├── slowfast/
│   └── datasets/
│       ├── coronary.py          # 冠脉 CT 数据集类
│       └── __init__.py          # 已更新，注册 Coronary 数据集
├── configs/
│   └── Coronary/
│       └── MVITv2_B_32x3.yaml   # MViTV2-B 配置文件
├── tools/
│   └── generate_coronary_splits.py  # 生成训练/验证/测试分割
└── CORONARY.md                  # 完整文档
```

## 2. 数据准备

运行分割生成脚本（已完成）：

```bash
cd E:\pycharm23\Projs\DcmDataset\SlowFast
python tools/generate_coronary_splits.py
```

生成的文件：
- `Central/train.csv` - 77 个训练样本
- `Central/val.csv` - 77 个验证样本
- `Central/test.csv` - 20 个测试样本

## 3. 配置修改

编辑 `configs/Coronary/MVITv2_B_32x3.yaml`，更新路径：

```yaml
DATA:
  PATH_TO_DATA_DIR: E:/pycharm23/Projs/DcmDataset/Central
  PATH_PREFIX: E:/pycharm23/Projs/DcmDataset/Central
```

**注意**: Windows 路径使用正斜杠 `/` 或双反斜杠 `\\`

## 4. 依赖安装

确保安装以下依赖：

```bash
pip install opencv-python simplejson pandas numpy torch torchvision
```

SlowFast 框架依赖：

```bash
cd SlowFast
pip install -e .
```

## 5. 训练命令

### 单机单卡训练

```bash
cd E:\pycharm23\Projs\DcmDataset\SlowFast

python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  NUM_GPUS 1 \
  DATA.PATH_TO_DATA_DIR E:/pycharm23/Projs/DcmDataset/Central \
  DATA.PATH_PREFIX E:/pycharm23/Projs/DcmDataset/Central
```

### 使用 PowerShell

```powershell
cd E:\pycharm23\Projs\DcmDataset\SlowFast

python tools/run_net.py `
  --cfg configs/Coronary/MVITv2_B_32x3.yaml `
  NUM_GPUS 1 `
  DATA.PATH_TO_DATA_DIR E:/pycharm23/Projs/DcmDataset/Central `
  DATA.PATH_PREFIX E:/pycharm23/Projs/DcmDataset/Central
```

## 6. 数据集工作流程

1. **加载元数据**: 从 `metadataV0.csv` 读取样本信息
2. **过滤分割**: 根据 `SplitA` 或 `SplitB` 列筛选训练/测试样本
3. **加载 DICOM**: 从 ID 命名的文件夹读取 DICOM 序列
4. **提取帧范围**: 使用 `Start Frame` 和 `End Frame` 确定可用范围
5. **应用窗宽窗位**: 使用心脏 CT 预设（宽 400，中心 50）
6. **时间采样**: 按 `NUM_FRAMES` 和 `SAMPLING_RATE` 采样帧
7. **数据增强**: 随机翻转、裁剪、缩放（训练时）
8. **返回样本**: `(frames, label, index, time_idx, metadata)`

## 7. 关键参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATA.NUM_FRAMES` | 32 | 每次采样的帧数 |
| `DATA.SAMPLING_RATE` | 2 | 帧采样间隔 |
| `DATA.TRAIN_CROP_SIZE` | 224 | 训练裁剪尺寸 |
| `DATA.SPLIT_COLUMN` | SplitA | 使用哪个分割列 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL.NUM_CLASSES` | 1 | 回归任务（单输出） |
| `MODEL.LOSS_FUNC` | mse | 均方误差损失 |
| `MVIT.DEPTH` | 24 | Transformer 层数 |
| `MVIT.EMBED_DIM` | 96 | 嵌入维度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SOLVER.BASE_LR` | 0.00005 | 学习率 |
| `SOLVER.MAX_EPOCH` | 100 | 最大训练轮数 |
| `TRAIN.BATCH_SIZE` | 4 | 每卡批次大小 |
| `TRAIN.EVAL_PERIOD` | 5 | 验证间隔 |

## 8. 输出内容

训练输出保存在 `outputs/coronary_mvitt2_b/`：

```
outputs/coronary_mvitt2_b/
├── config.yaml           # 使用的配置
├── log.txt              # 训练日志
├── checkpoint_epoch_005.pkl
├── checkpoint_epoch_010.pkl
├── ...
└── runs/                # TensorBoard 日志
```

查看训练日志：

```bash
tail -f outputs/coronary_mvitt2_b/log.txt
```

## 9. 常见问题

### Q: CUDA 内存不足

**A**: 减小批次大小或帧数：
```yaml
TRAIN.BATCH_SIZE: 2
DATA.NUM_FRAMES: 16
```

### Q: 加载 DICOM 失败

**A**: 检查：
- 患者 ID 文件夹是否存在
- DICOM 文件是否完整
- `Start Frame` 和 `End Frame` 是否超出范围

### Q: 训练损失不下降

**A**: 尝试：
- 增加学习率 `SOLVER.BASE_LR: 0.0001`
- 增加训练轮数 `SOLVER.MAX_EPOCH: 200`
- 检查标签值是否正确

## 10. 使用 SplitB

如需使用 SplitB 分割：

```bash
python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  DATA.SPLIT_COLUMN SplitB \
  NUM_GPUS 1
```

## 11. 验证/测试

训练完成后，使用最佳模型进行测试：

```bash
python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  TEST.ENABLE True \
  TEST.CHECKPOINT_PATH outputs/coronary_mvitt2_b/checkpoint_best.pkl \
  NUM_GPUS 1
```

## 12. 代码验证

测试数据集加载：

```python
import sys
sys.path.insert(0, r'E:\pycharm23\Projs\DcmDataset')
sys.path.insert(0, r'E:\pycharm23\Projs\DcmDataset\SlowFast')

from dcm import read_series, apply_window_preset

# 测试 DICOM 加载
series = read_series(r'E:\pycharm23\Projs\DcmDataset\Central\163329')
print(f"Loaded {len(series)} frames")

# 测试体积转换
volume = series.to_volume()
print(f"Volume shape: {volume.shape}")

# 测试窗宽窗位
windowed = apply_window_preset(volume, 'cardiac')
print(f"Windowed range: [{windowed.min():.2f}, {windowed.max():.2f}]")
```
