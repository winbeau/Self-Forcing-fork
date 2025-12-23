# 注意力可视化逻辑

本文档描述 `notebooks/extract_all_attention.ipynb` 的绘图逻辑。

## 概述

该 notebook 生成两张图来分析单层的注意力分布：

1. **2D 热力图**: Query Frame × Key Frame，3×4 网格显示 12 个 head
2. **Per-Head Grid**: 最后一个 block 对各帧的注意力柱状图

## 数据加载

```python
import torch

DATA_PATH = "../cache/layer3.pt"
data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

# 完整的 frame×frame 注意力矩阵
full_frame_attn = data['full_frame_attention'].float().numpy()  # [12, 19, 19]

# 最后一个 block 的帧注意力
last_block_attn = data['last_block_frame_attention'].float().numpy()  # [12, 19]
```

## 图 1: 2D 热力图

### 布局

- 3×4 网格，共 12 个子图（每个 head 一张）
- X 轴: Key Frame Index
- Y 轴: Query Frame Index
- 颜色: RdBu_r colormap（红=正值，蓝=负值）

### 关键代码

```python
import matplotlib.pyplot as plt

ncols = 4
nrows = 3  # ceil(12 / 4)

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
axes = axes.flatten()

# 对称的颜色范围（便于 RdBu_r 以 0 为中心）
vmax = max(abs(full_frame_attn.min()), abs(full_frame_attn.max()))
vmin = -vmax

for h in range(num_heads):
    ax = axes[h]
    attn_map = full_frame_attn[h]  # [Q, K]

    im = ax.imshow(
        attn_map,
        cmap="RdBu_r",      # 红蓝发散色板
        aspect="auto",
        origin="lower",      # Y 轴从下到上
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    ax.set_title(f"Head {h}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Key Frame", fontsize=9)
    ax.set_ylabel("Query Frame", fontsize=9)

# 全局 colorbar
fig.colorbar(im, cax=cbar_ax, label="Attention Logits")
```

### 输出示例

```
Layer 3: 2D Attention Maps (All Heads)
┌────────┬────────┬────────┬────────┐
│ Head 0 │ Head 1 │ Head 2 │ Head 3 │
├────────┼────────┼────────┼────────┤
│ Head 4 │ Head 5 │ Head 6 │ Head 7 │
├────────┼────────┼────────┼────────┤
│ Head 8 │ Head 9 │ Head10 │ Head11 │
└────────┴────────┴────────┴────────┴──[colorbar]
```

## 图 2: Per-Head Grid 柱状图

### 布局

- 3×4 网格，共 12 个子图
- X 轴: Key Frame Index (0-18)
- Y 轴: Attention Logits
- 每个子图显示最后一个 block (frames 16-18) 对所有帧的平均注意力

### Sink Score 计算

```python
# Sink score = 首帧注意力 - 中间帧平均注意力
first = head[0]
middle = head[1:-1].mean()
sink_score = first - middle
```

Sink score > 0 表示该 head 存在 attention sink 现象（首帧获得异常高的注意力）。

### 关键代码

```python
import seaborn as sns

BAR_COLOR = sns.color_palette("colorblind")[0]  # 蓝色

for h in range(num_heads):
    ax = axes[h]
    head = last_block_attn[h]  # [K]

    # 计算 sink score
    first = head[0]
    middle = head[1:-1].mean()
    sink_score = first - middle

    # 柱状图 + 折线图
    ax.bar(key_indices, head, alpha=0.85, width=0.8, color=BAR_COLOR)
    ax.plot(key_indices, head, "o-", color="black", linewidth=1, markersize=2)

    ax.set_title(f"H{h} (sink={sink_score:.2f})", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
```

### 输出示例

```
Layer 3: Per-Head Attention Distribution
Query: frames [16, 17, 18]

┌──────────────┬──────────────┬──────────────┬──────────────┐
│ H0 (sink=7.1)│ H1 (sink=0.5)│ H2 (sink=-0.6)│ H3 (sink=0.5)│
│   ▂▁▁▁▁▁▁▂   │   ▁▁▁▁▁▁▁▂   │   ▁▁▁▁▁▁▁▂    │   ▁▁▁▁▁▁▁▂   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ H4 (sink=2.6)│ H5 (sink=0.2)│ H6 (sink=1.6)│ H7 (sink=1.3)│
│   ▃▁▁▁▁▁▁▄   │   ▄▄▄▄▄▄▄▅   │   ▅▄▄▄▄▄▄█    │   ▁▁▁▁▁▁▁▄   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ H8 (sink=0.1)│ H9 (sink=1.7)│ H10(sink=0.5)│ H11(sink=0.2)│
│   ▁▁▁▁▁▁▁▂   │   ▃▁▁▁▁▁▁▅   │   ▁▁▁▁▁▁▁▂    │   ▁▁▁▁▁▁▁▂   │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## 颜色主题

### 热力图

使用 **RdBu_r** (Red-Blue reversed) 发散色板：
- 红色: 正值（高注意力）
- 白色: 零值
- 蓝色: 负值（低注意力）

```python
cmap = "RdBu_r"
# 或自定义色板
hex_colors = [
    "#98202C", "#D15C4C", "#EEA17E", "#FDDAC3",
    "#F0F4F5",  # 中间过渡色
    "#C2DEEC", "#79B0D4", "#3C7FB9", "#1D4680"
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", hex_colors)
```

### 柱状图

使用 seaborn colorblind 调色板的第一个颜色（蓝色）：

```python
BAR_COLOR = sns.color_palette("colorblind")[0]
```

## 统计输出

```python
# 对角线平均（self-attention 强度）
diag_mean = np.diagonal(full_frame_attn, axis1=1, axis2=2).mean()

# 首帧平均（sink 强度）
first_col_mean = full_frame_attn[:, :, 0].mean()

# Per-head 统计
for h in range(num_heads):
    head = last_block_attn[h]
    first = head[0]
    middle = head[1:-1].mean()
    last = head[-1]
    sink = first - middle
    print(f"H{h:2d}: first={first:.3f}, mid={middle:.3f}, last={last:.3f}, sink={sink:+.3f}")
```

## 保存格式

```python
SAVE_SVG = True
SAVE_DIR = "attention_analysis/layer3"

if SAVE_SVG:
    plt.savefig(f"{SAVE_DIR}/layer{idx}_2d_heatmap.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{SAVE_DIR}/layer{idx}_perhead_grid.svg", format="svg", bbox_inches="tight")
```

## Matplotlib 配置

```python
plt.rcParams.update({
    "svg.fonttype": "none",          # 保持文本可编辑
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
```

## 使用方法

1. 运行提取脚本生成数据：
   ```bash
   PYTHONPATH=. python experiments/run_extraction_each.py \
       --layer_index 3 \
       --output_path cache/layer3.pt
   ```

2. 修改 notebook 中的配置：
   ```python
   DATA_PATH = "../cache/layer3.pt"
   SAVE_DIR = "attention_analysis/layer3"
   ```

3. 运行所有 cell 生成图表

## 相关文件

- `notebooks/extract_all_attention.ipynb` - 可视化 notebook
- `experiments/run_extraction_each.py` - 数据提取脚本
- `docs/attention_extraction.md` - 提取逻辑文档
