# 冠状动脉 CT 多任务学习使用说明

本文档说明了如何在 SlowFast 框架下进行冠状动脉 CT 视频的分类和回归联合训练。

## 目录结构

```
SlowFast/
├── configs/Coronary/
│   └── MVITv2_B_32x3_multitask.yaml   # 多任务配置文件
├── slowfast/
│   ├── models/
│   │   ├── coronary_head.py           # 多任务头模块
│   │   └── coronary_loss.py           # 多任务损失函数
│   ├── datasets/
│   │   └── coronary.py                # 数据集加载器 (已扩展)
│   ├── config/
│   │   └── defaults.py                # 配置定义 (已扩展)
│   └── tools/
│       └── train_coronary_multitask.py # 训练脚本
└── run_coronary_training.py           # 启动脚本
```

## 功能特性

### 1. 多任务学习
- **分类任务**: 预测斑块存在的置信度 (0-1)
- **回归任务**: 预测归一化的斑块堵塞程度 (0-1)

### 2. 多 Token 输出 (可选)
- 从 class token 生成多个 proposal token
- 每个 token 对应一个独立的预测
- 支持可数量的输出 token 配置

### 3. 参数化配置
所有关键参数都可在配置文件中修改:
- `NUM_PROPOSALS`: 输出 token 数量 (默认 10)
- `CONFIDENCE_THRESHOLD`: 置信度阈值 (默认 0.5)
- `CLS_HIDDEN_DIM`: 分类头隐藏层维度
- `REG_HIDDEN_DIM`: 回归头隐藏层维度
- `CLS_LOSS_WEIGHT`: 分类损失权重
- `REG_LOSS_WEIGHT`: 回归损失权重
- `PLAQUE_NORM_FACTOR`: 斑块归一化因子

## 快速开始

### 在本地运行训练

```bash
cd E:\pycharm23\Projs\DcmDataset\SlowFast

# 使用默认配置训练
python run_coronary_training.py

# 或直接用 SlowFast 命令
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml
```

### 在远程服务器上运行

```bash
# 1. 上传代码到远程服务器
python run_coronary_training.py --upload

# 2. 在远程服务器上运行训练
python run_coronary_training.py --remote
```

### 使用 SSH 工具直接执行

```python
from utils.ssh_exec import SSHClient

with SSHClient() as ssh:
    # 上传文件
    ssh.upload_file("local/path/file.py", "/remote/path/file.py")

    # 执行训练命令
    ssh.exec_command(
        "cd /18018998051/CTA && "
        "python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_multitask.yaml"
    )
```

## 配置文件说明

### 关键配置项

```yaml
CORONARY:
  # 多 token 输出设置
  USE_MULTI_TOKEN: True          # 是否使用多 token 输出
  NUM_PROPOSALS: 10              # proposal token 数量
  CONFIDENCE_THRESHOLD: 0.5      # 测试时置信度阈值

  # 网络结构
  CLS_HIDDEN_DIM: 256            # 分类头隐藏层维度
  REG_HIDDEN_DIM: 256            # 回归头隐藏层维度

  # 损失权重
  CLS_LOSS_WEIGHT: 1.0           # 分类损失权重
  REG_LOSS_WEIGHT: 1.0           # 回归损失权重

  # 数据归一化
  PLAQUE_NORM_FACTOR: 100.0      # 斑块归一化因子
```

## 数据要求

### 输入数据格式

1. **DICOM 序列**: 按 ID 组织的文件夹结构
   ```
   /18018998051/data/Central/CTA/
   ├── 870328/
   ├── 828085/
   └── ...
   ```

2. **metadata 文件**: `metadataV1.csv` 包含以下列:
   - `ID`: 样本 ID
   - `Plaque`: 斑块百分比值
   - `Branch`: 血管分支类型
   - `Split`: 数据划分 (Train/Val/Test)

## 模型输出

### 训练时输出
```python
{
    'cls_outputs': [tensor1, tensor2, ...],  # 分类输出列表
    'reg_outputs': [tensor1, tensor2, ...],  # 回归输出列表
    'all_tokens': tensor                      # 所有 proposal tokens
}
```

### 测试时处理
```python
# 获取预测
outputs = model(inputs)

# 多 token 模式：平均所有 proposal 的预测
if USE_MULTI_TOKEN:
    cls_pred = torch.stack(outputs['cls_outputs']).mean(0)
    reg_pred = torch.stack(outputs['reg_outputs']).mean(0)
else:
    cls_pred = outputs['cls_outputs'][0]
    reg_pred = outputs['reg_outputs'][0]

# 应用置信度阈值
predictions = []
for i, conf in enumerate(cls_pred):
    if conf >= CONFIDENCE_THRESHOLD:
        predictions.append({
            'confidence': conf.item(),
            'plaque': reg_pred[i].item()
        })
```

## 评估指标

训练过程中会记录以下指标:

### 分类指标
- **Accuracy**: 分类准确率 (使用置信度阈值)

### 回归指标
- **MAE**: 平均绝对误差
- **MSE**: 均方误差

## 常见问题

### Q: 如何修改输出 token 数量？
A: 在配置文件中修改 `CORONARY.NUM_PROPOSALS`

### Q: 如何调整置信度阈值？
A: 在配置文件中修改 `CORONARY.CONFIDENCE_THRESHOLD`

### Q: 如何改变损失权重比例？
A: 修改 `CORONARY.CLS_LOSS_WEIGHT` 和 `CORONARY.REG_LOSS_WEIGHT`

### Q: 如何切换到单 token 模式？
A: 设置 `CORONARY.USE_MULTI_TOKEN: False`

## 远程服务器配置

SSH 连接信息存储在 `E:\pycharm23\Projs\DcmDataset\remote_config.json`:

```json
{
    "ssh": {
        "host": "10.8.131.51",
        "port": 30312,
        "user": "root",
        "password": "your_password"
    },
    "paths": {
        "remote_base": "/18018998051/CTA",
        "local_base": "E:/pycharm23/Projs/DcmDataset"
    },
    "conda": {
        "env_name": "mamba"
    }
}
```

## 参考资料

- [SlowFast 官方文档](https://github.com/facebookresearch/SlowFast)
- [MViT 论文](https://arxiv.org/abs/2112.01526)
- [冠状动脉数据集说明](./CORONARY.md)
