# 匈牙利损失（Hungarian Loss）用于冠脉多类别分类 + 回归任务

## 概述

针对你提出的**多标签回归 + 分类任务**，我们可以将二分图匹配逻辑严谨地改写为以下算法形式。

在这种设定下，每个预测槽输出：
- **分类**: 5 类概率分布（4 个前景类：LAD, RCA, LCX, LM + 1 个背景类）
- **回归**: 一个标量值 $v_i$（斑块狭窄程度，0-1）

---

### 1. 符号定义 (Problem Formulation)

*   **真值集合 ($y$)**：令 $y = \{y_i\}_{i=1}^N$ 为大小为 $N$ 的真值集合，其中包含 $M$ 个真实标注和 $N-M$ 个空值 $\emptyset$。
    *   每个真值 $y_i = (c_i, v_i)$，其中 $c_i \in \{0,1,2,3,4\}$ 为目标类别标签（0-3 为血管分支类型，4 为背景），$v_i \in \mathbb{R}$ 为回归的目标单一数值。
*   **预测集合 ($\hat{y}$)**：令 $\hat{y} = \{\hat{y}_j\}_{j=1}^N$ 为模型输出的 $N$ 个预测结果。
    *   每个预测 $\hat{y}_j = (\hat{p}_j(c), \hat{v}_j)$，其中 $\hat{p}_j(c)$ 是预测类别 $c$ 的概率分布（经过 softmax），$\hat{v}_j \in \mathbb{R}$ 是预测的回归标量值。

---

### 2. 匹配代价函数 (Matching Cost)

为了寻找最优的一对一匹配，我们定义配对代价函数 $\mathcal{L}_{match}$。对于第 $i$ 个真值和索引为 $\sigma(i)$ 的预测值，其匹配代价为：

$$\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)}) = - \mathbb{1}_{\{c_i \neq \emptyset\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} \lambda_{reg} \mathcal{L}_{reg}(v_i, \hat{v}_{\sigma(i)})$$

**说明：**
*   **分类项**：使用预测正确类别的**概率** $\hat{p}_{\sigma(i)}(c_i)$（经过 softmax），这符合 DETR 的经典做法，旨在让分类项与回归项在量级上更易平衡。
*   **回归项 ($\mathcal{L}_{reg}$)**：由于预测是单一数值，通常采用 **$L_2$ 损失**即**均方误差 (MSE)**。
*   **权重 ($\lambda_{reg}$)**：用于调节回归任务在匹配过程中的重要程度。

---

### 3. 匈牙利算法求解 (Optimal Assignment)

寻找一个最优排列 $\hat{\sigma} \in \mathfrak{S}_N$，使得总配对代价最小：

$$\hat{\sigma} = \arg \min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})$$

该过程通过**匈牙利算法 (Hungarian Algorithm)** 实现，确保每个真值（包括空值）都被唯一地分配给一个预测槽。

---

### 4. 最终训练损失 (Hungarian Loss)

一旦确定了最优匹配 $\hat{\sigma}$，模型根据该分配关系计算总损失：

$$\mathcal{L}_{Hungarian}(y, \hat{y}) = \sum_{i=1}^N \left[ \mathcal{L}_{cls}(c_i, \hat{p}_{\hat{\sigma}(i)}) + \mathbb{1}_{\{c_i \neq \emptyset\}} \lambda_{reg} \mathcal{L}_{reg}(v_i, \hat{v}_{\hat{\sigma}(i)}) \right]$$

**关键改动点：**
1.  **分类损失 ($\mathcal{L}_{cls}$)**：采用**交叉熵损失**（CrossEntropyLoss），支持 5 类分类（4 个前景类 + 1 个背景类）。
2.  **回归损失**：仅针对非空匹配项（前景类）计算预测值 $\hat{v}_{\hat{\sigma}(i)}$ 与目标值 $v_i$ 之间的回归偏差。
3.  **背景处理**：
    *   对于匹配到背景类（$c_i = 4$）的目标，仅计算分类损失。
    *   对于匹配到 $\emptyset$（padding）的预测槽，仅计算分类损失（将其分类为背景）。

---

## 类别定义

| 类别索引 | 含义 | 说明 |
|---------|------|------|
| 0 | LAD | 左前降支 |
| 1 | RCA | 右冠状动脉 |
| 2 | LCX | 左回旋支 |
| 3 | LM | 左主干 |
| 4 | Background | 背景（无斑块） |
| -1 | Padding | 填充（损失计算时忽略） |

---

## 数据格式

### 分类目标 (cls_targets)

```python
# 形状：[batch, num_proposals]
# dtype: torch.long
# 值：0-3 为前景类，4 为背景，-1 为 padding

# 示例（num_proposals=5）：
# 样本 1: [0, 1, 4, -1, -1]  # 2 个真实目标（LAD, RCA），1 个背景，2 个 padding
# 样本 2: [2, 3, 1, 4, -1]   # 3 个真实目标，1 个背景，1 个 padding
```

### 回归目标 (reg_targets)

```python
# 形状：[batch, num_proposals]
# dtype: torch.float32
# 值：[0, 1] 范围内的斑块狭窄程度

# 示例（num_proposals=5）：
# 样本 1: [0.45, 0.72, 0.0, 0.0, 0.0]  # padding 位置为 0
# 样本 2: [0.33, 0.88, 0.15, 0.0, 0.0]
```

### 有效掩码 (valid_mask)

```python
# 形状：[batch, num_proposals]
# dtype: torch.float32
# 值：1 表示真实目标或背景，0 表示 padding

# 示例（num_proposals=5）：
# 样本 1: [1.0, 1.0, 1.0, 0.0, 0.0]  # 前 3 个有效，后 2 个 padding
# 样本 2: [1.0, 1.0, 1.0, 1.0, 0.0]
```

---

## 模型输出

```python
outputs = {
    'cls_outputs': [
        tensor1,  # [batch, 5] - 类别 logits（5 类）
        tensor2,  # [batch, 5]
        ...
    ],  # List 长度 = num_proposals

    'reg_outputs': [
        tensor1,  # [batch, 1] - 回归值（0-1）
        tensor2,  # [batch, 1]
        ...
    ],  # List 长度 = num_proposals
}
```

---

## 训练命令

```bash
# 使用匈牙利损失训练（多类别分类 + 回归）
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml

# 调整预测槽数量
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml \
    CORONARY.NUM_PROPOSALS 15

# 调整匹配代价权重
python tools/run_net.py --cfg configs/Coronary/MVITv2_B_32x3_hungarian.yaml \
    HUNGARIAN.COST_REG_WEIGHT 2.0
```

---

## 实现细节

### 分类头结构

```python
# 每个 proposal 的分类头
cls_head = nn.Sequential(
    nn.Linear(dim_in, 256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, num_classes)  # 输出 5 类 logits（无 Sigmoid/Softmax）
)
```

### 回归头结构

```python
# 每个 proposal 的回归头
reg_head = nn.Sequential(
    nn.Linear(dim_in, 256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1),
    nn.Sigmoid()  # 输出范围 [0, 1]
)
```

### 损失计算流程

1. **匹配阶段**：
   - 对分类 logits 应用 softmax 得到概率
   - 计算匹配代价矩阵（分类概率 + 回归距离）
   - 使用匈牙利算法求解最优匹配

2. **损失计算阶段**：
   - 对匹配的 foreground 目标：计算 CrossEntropyLoss + MSELoss
   - 对匹配的 background 目标：仅计算 CrossEntropyLoss
   - 对未匹配的预测：仅计算 CrossEntropyLoss（目标为背景类）
