# 匈牙利损失（Hungarian Loss）使用说明

## 概述

本项目实现了基于匈牙利匹配的集合预测损失函数，用于冠状动脉 CT 血管造影的多任务学习（分类 + 回归）。

### 问题背景

传统的多标签训练方法直接预测多个结果并进行损失计算，这会导致以下问题：
1. **预测顺序敏感**: 模型需要学习预测的固定顺序
2. **重复预测**: 多个预测可能收敛到相同的目标
3. **匹配模糊**: 无法确定哪个预测应该对应哪个目标

### 解决方案

将问题重塑为**集合预测问题**，使用匈牙利算法寻找预测和目标之间的最优一对一匹配：
1. 每个预测槽输出独立的预测
2. 使用匈牙利算法找到最优匹配
3. 仅对匹配的对计算损失

---

## 算法原理

### 1. 符号定义

- **真值集合**: $y = \{y_i\}_{i=1}^N$，包含 $M$ 个真实标注和 $N-M$ 个空值
  - 每个真值 $y_i = (c_i, v_i)$，其中 $c_i$ 为分类标签，$v_i$ 为回归值

- **预测集合**: $\hat{y} = \{\hat{y}_j\}_{j=1}^N$
  - 每个预测 $\hat{y}_j = (\hat{p}_j, \hat{v}_j)$，其中 $\hat{p}_j$ 是分类概率，$\hat{v}_j$ 是回归预测值

### 2. 匹配代价函数

对于第 $i$ 个真值和索引为 $\sigma(i)$ 的预测，匹配代价为：

$$\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)}) = -\hat{p}_{\sigma(i)}(c_i) + \lambda_{reg} \cdot \mathcal{L}_{reg}(v_i, \hat{v}_{\sigma(i)})$$

其中：
- **分类项**: 使用预测概率（负值，概率越高代价越低）
- **回归项**: 使用 MSE 或 L1 损失
- $\lambda_{reg}$: 回归权重

### 3. 匈牙利算法求解

寻找最优排列 $\hat{\sigma}$ 使得总代价最小：

$$\hat{\sigma} = \arg \min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})$$

### 4. 最终损失

基于最优匹配计算损失：

$$\mathcal{L} = \sum_{i=1}^N \left[ \mathcal{L}_{cls}(c_i, \hat{p}_{\hat{\sigma}(i)}) + \mathbb{1}_{\{c_i > 0\}} \cdot \lambda_{reg} \cdot \mathcal{L}_{reg}(v_i, \hat{v}_{\hat{\sigma}(i)}) \right]$$

---

## 代码结构

```
slowfast/models/
├── hungarian_loss.py          # 匈牙利损失实现
│   ├── HungarianMatcher       # 匈牙利匹配器
│   ├── HungarianLoss          # 基础版本
│   ├── HungarianLossV2        # 改进版本（推荐）
│   ├── FocalLoss              # Focal 损失（可选）
│   └── SmoothL1Loss           # Smooth L1 损失（可选）
└── coronary_loss.py           # 原始多任务损失（对比参考）
```

### HungarianMatcher

```python
matcher = HungarianMatcher(
    cost_cls_weight=1.0,      # 分类代价权重
    cost_reg_weight=1.0,      # 回归代价权重
    cost_reg_type='mse',      # 'mse' 或 'l1'
)

matches = matcher(
    cls_outputs=cls_outputs,  # List of [B, 1]
    reg_outputs=reg_outputs,  # List of [B, 1]
    cls_targets=cls_targets,  # [B, N]
    reg_targets=reg_targets,  # [B, N]
    valid_mask=valid_mask,    # [B, N], optional
)
# 返回：List of (pred_indices, target_indices)
```

### HungarianLossV2（推荐）

改进的特性：
1. 支持 Focal Loss 处理类别不平衡
2. 支持 Smooth L1 Loss 提高鲁棒性
3. 更好地处理全背景样本

---

## 配置方法

### 1. 基础配置

在 YAML 配置文件中设置：

```yaml
CORONARY:
  USE_MULTI_TOKEN: True
  NUM_PROPOSALS: 10              # 预测槽数量
  LOSS_TYPE: 'hungarian'         # 使用匈牙利损失
  HUNGARIAN_LOSS_VERSION: 'v2'   # 使用改进版本
  CLS_LOSS_WEIGHT: 1.0
  REG_LOSS_WEIGHT: 1.0

HUNGARIAN:
  COST_CLS_WEIGHT: 1.0           # 匹配时的分类代价权重
  COST_REG_WEIGHT: 1.0           # 匹配时的回归代价权重
  COST_REG_TYPE: 'mse'           # 'mse' 或 'l1'
```

### 2. 可选配置

```yaml
CORONARY:
  # 使用 Focal Loss（处理类别不平衡）
  LOSS_TYPE: 'focal'
  FOCAL_ALPHA: 0.25
  FOCAL_GAMMA: 2.0

  # 使用 Smooth L1 Loss（鲁棒回归）
  LOSS_TYPE: 'smooth_l1'
  SMOOTH_L1_BETA: 0.1
```

---

## 使用方法

### 训练命令

```bash
# 使用匈牙利损失训练
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml

# 调整预测槽数量
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml \
    CORONARY.NUM_PROPOSALS 20

# 调整匹配代价权重
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml \
    HUNGARIAN.COST_REG_WEIGHT 2.0
```

### 数据要求

数据集需要返回：
- `cls_targets`: [batch, num_proposals] 分类目标
- `reg_targets`: [batch, num_proposals] 回归目标
- `valid_mask`: [batch, num_proposals] 有效掩码（1=真实目标，0=padding）

### 模型输出

模型需要输出：
- `cls_outputs`: List of [batch, 1] 分类预测
- `reg_outputs`: List of [batch, 1] 回归预测

---

## 训练技巧

### 1. 预测槽数量设置

- `NUM_PROPOSALS` 应略大于数据集平均目标数量
- 例如：平均每个样本有 4-6 个斑块，设置 `NUM_PROPOSALS=10`

### 2. 代价权重调整

如果模型倾向于：
- **漏检**: 增加 `HUNGARIAN.COST_CLS_WEIGHT`
- **重复预测**: 增加 `HUNGARIAN.COST_REG_WEIGHT`

### 3. 损失权重平衡

如果分类/回归任务收敛速度差异大：
- 分类收敛慢：增加 `CORONARY.CLS_LOSS_WEIGHT`
- 回归收敛慢：增加 `CORONARY.REG_LOSS_WEIGHT`

---

## 与原始方法的对比

| 特性 | 原始方法 | 匈牙利匹配 |
|------|----------|------------|
| 预测顺序 | 固定顺序 | 无序集合 |
| 重复预测 | 可能发生 | 不会发生 |
| 匹配方式 | 位置对应 | 最优匹配 |
| 背景处理 | 需要显式 padding | 自动处理 |

---

## 常见问题

### Q1: 为什么需要匈牙利匹配？

A: 传统方法假设预测和目标按固定顺序对应，这迫使模型学习任意排序。匈牙利匹配允许模型以无序方式预测，更符合检测任务的本质。

### Q2: 如何设置预测槽数量？

A: 设置为略大于数据集中最大目标数量。过多的预测槽会增加计算量，过少会导致目标无法匹配。

### Q3: 训练不稳定怎么办？

A: 尝试：
1. 增加 warmup epochs
2. 降低初始学习率
3. 调整 `COST_CLS_WEIGHT` 和 `COST_REG_WEIGHT` 的比例

### Q4: 可以使用其他回归损失吗？

A: 可以。在配置中设置 `LOSS_TYPE='smooth_l1'` 使用 Smooth L1 损失，或设置 `HUNGARIAN.COST_REG_TYPE='l1'` 使用 L1 代价。

---

## 参考资料

- DETR: End-to-End Object Detection with Transformers (ECCV 2020)
- Hungarian Algorithm: Harold W. Kuhn, "The Hungarian Method for the Assignment Problem"
