# plt_fig4.ipynb 详解

## 概述

本 Jupyter Notebook 用于可视化注意力权重数据，复现 Deep Forcing 论文的 Figure 4（Attention Sink 现象）。

**输入**：`run_extraction_figure4.py` 生成的 `.pt` 文件
**输出**：多个 SVG 格式的可视化图表

## 文件位置

```
notebooks/plt_fig4.ipynb
```

## 生成的图表

| 文件名 | 内容 |
|--------|------|
| `figure4_reproduction.svg` | 主图：两个代表性 head 的帧间注意力分布 |
| `figure4_layer1_per_head_grid.svg` | Layer 1 所有 head 的网格图 |
| `figure4_layer5_per_head_grid.svg` | Layer 5 所有 head 的网格图 |
| `figure4_layer1_all_heads.svg` | Layer 1 热力图 |
| `figure4_layer5_all_heads.svg` | Layer 5 热力图 |
| `figure4_layer_comparison.svg` | 跨层对比图 |

---

## Cell 1: 环境设置

### 代码

```python
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# 设置绘图风格
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

plt.rcParams.update({
    "svg.fonttype": "none",           # SVG 中使用真实字体而非路径
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
```

### 关键设置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `svg.fonttype` | `"none"` | SVG 保留可编辑文本（非曲线路径） |
| `figure.dpi` | `150` | 屏幕显示分辨率 |
| `savefig.dpi` | `300` | 保存时的高分辨率 |
| `savefig.bbox` | `"tight"` | 自动裁剪空白边距 |

---

## Cell 2: 加载数据

### 代码

```python
# 加载数据
data = torch.load("../cache/attention_cache_wan_base.pt", map_location="cpu", weights_only=False)

print("=" * 70)
print("加载注意力数据")
print("=" * 70)
print(f'Prompt: {data.get("prompt", "N/A")}')
print(f'总帧数: {data.get("num_frames", "N/A")}')
print(f'每帧 token 数: {data.get("frame_seq_length", "N/A")}')
print(f'Query 帧: {data.get("query_frames", "N/A")}')
print(f'Key 帧: {data.get("key_frames", "N/A")}')

attention_weights = data["attention_weights"]
frame_seq_length = data.get("frame_seq_length", 1560)

for i, w in enumerate(attention_weights):
    layer_idx = w["layer_idx"]
    attn = w["attn_logits"]
    num_key_frames = w.get("num_key_frames", attn.shape[3] // frame_seq_length)
    print(f"Layer {layer_idx}: shape={tuple(attn.shape)}, {num_key_frames} key frames")
    print(f"  Range: [{attn.float().min().item():.4f}, {attn.float().max().item():.4f}]")
```

### 数据结构

```python
data = {
    'attention_weights': [
        {
            'layer_idx': 0,
            'attn_logits': Tensor[1, 12, 4680, 32760],  # [B, N, Q, K]
            'is_logits': True,
            'num_frames': 21,
            'frame_seq_length': 1560,
            'num_heads': 12,
            'num_query_frames': 3,
            'num_key_frames': 21,
        },
        {
            'layer_idx': 4,
            ...
        }
    ],
    'prompt': "A majestic eagle...",
    'num_frames': 21,
    'frame_seq_length': 1560,
    'query_frames': [18, 19, 20],
    'key_frames': [0, 1, 2, ..., 20],
}
```

### 输出示例

```
======================================================================
加载注意力数据
======================================================================
Prompt: A majestic eagle soaring through a cloudy sky, cinematic lighting
总帧数: 21
每帧 token 数: 1560
Query 帧: [18, 19, 20]
Key 帧: [0, 1, 2, ..., 20]

Layer 0: shape=(1, 12, 4680, 32760), 21 key frames
  Range: [-7.8125, 23.5000]
Layer 4: shape=(1, 12, 4680, 32760), 21 key frames
  Range: [-28.5000, 50.2500]
```

**注意**：Range 包含负值，说明这是 pre-softmax logits（不是概率）。

---

## Cell 3: 计算帧间注意力分布（核心函数）

### 代码

```python
def compute_frame_attention(attn_data, frame_seq_length=1560):
    """
    将 token 级别的 attention 聚合为帧级别。

    输入: attn_logits [B, N, Lq, Lk]
    输出: frame_attention [N, num_key_frames]
    """
    attn = attn_data["attn_logits"][0].float().numpy()  # [num_heads, Lq, Lk]
    num_heads, lq, lk = attn.shape
    fsl = attn_data.get("frame_seq_length", frame_seq_length)
    num_key_frames = lk // fsl

    frame_attention = np.zeros((num_heads, num_key_frames))

    for h in range(num_heads):
        head_attn = attn[h]  # [Lq, Lk]

        # 对所有 Query token 取平均
        avg_per_key = head_attn.mean(axis=0)  # [Lk]

        # 对每个 Key 帧内的 token 取平均
        for kf in range(num_key_frames):
            k_start = kf * fsl
            k_end = (kf + 1) * fsl
            frame_attention[h, kf] = avg_per_key[k_start:k_end].mean()

    return frame_attention, np.arange(num_key_frames)
```

### 聚合逻辑图解

```
原始 attention: [12 heads, 4680 query_tokens, 32760 key_tokens]

Step 1: 对 Query 维度取平均
        [12, 4680, 32760] → [12, 32760]
        含义：每个 head 的「平均 Query」对所有 Key token 的注意力

Step 2: 对每帧的 Key token 取平均
        [12, 32760] → [12, 21]
        含义：每个 head 对每个 Key 帧的平均注意力

最终: frame_attention[h, f] = head h 对 frame f 的平均注意力（logits）
```

### 为什么这样聚合

1. **Query 平均**：我们关心「整体」对历史帧的注意力，而非单个 token
2. **帧内平均**：同一帧内的 token 语义相近，取平均更稳定
3. **保留 Head 维度**：不同 head 可能有不同模式（如 sink head vs normal head）

---

## Cell 4: 处理数据

### 代码

```python
layer_attention_data = {}

for w in attention_weights:
    layer_idx = w["layer_idx"]
    frame_attn, key_indices = compute_frame_attention(w, frame_seq_length)

    layer_attention_data[layer_idx] = {
        "frame_attention": frame_attn,      # [num_heads, num_key_frames]
        "key_frame_indices": key_indices,   # [0, 1, 2, ..., 20]
        "num_heads": frame_attn.shape[0],
    }

    print(f"Layer {layer_idx}: {len(key_indices)} key frames, {frame_attn.shape[0]} heads")

    # 打印首尾帧差异
    for h in [0, frame_attn.shape[0] - 1]:
        head = frame_attn[h]
        diff = (head.max() - head.min()) / np.abs(head.mean()) * 100
        print(f"  Head {h+1}: first={head[0]:.4f}, last={head[-1]:.4f}, diff={diff:.1f}%")
```

### 输出示例

```
Layer 0: 21 key frames, 12 heads
  Head 1: first=1.9868, last=2.2374, diff=12.8%
  Head 12: first=5.3826, last=5.6694, diff=20.8%
Layer 4: 21 key frames, 12 heads
  Head 1: first=2.3686, last=2.9573, diff=21.8%
  Head 12: first=4.3439, last=5.4882, diff=25.8%
```

**diff 指标**：`(max - min) / |mean| * 100%`
- 值越大，帧间差异越明显
- Attention Sink 现象表现为 first frame 值显著高于其他帧

---

## Cell 5: 绘制主图（Figure 4 Reproduction）

### 代码

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#2E86AB", "#A23B72"]

# 选择代表性的两个 head
heads_to_show = [
    (0, 0, "L1H1"),   # Layer 1, Head 1
    (4, 9, "L5H10"),  # Layer 5, Head 10
]

for idx, (layer_idx, head_idx, label) in enumerate(heads_to_show):
    ax = axes[idx]
    if layer_idx not in layer_attention_data:
        continue

    frame_attn = layer_attention_data[layer_idx]["frame_attention"][head_idx]
    key_indices = layer_attention_data[layer_idx]["key_frame_indices"]
    color = colors[idx]

    # 柱状图 + 折线图
    ax.bar(key_indices, frame_attn, color=color, alpha=0.7, width=0.8)
    ax.plot(key_indices, frame_attn, "o-", color=color, linewidth=2, markersize=5)

    ax.set_xlabel("Key Frame Index", fontsize=12)
    ax.set_ylabel("Attention Weight (Logits)", fontsize=12)
    ax.set_title(f"{label} (Layer {layer_idx+1}, Head {head_idx+1})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # 标注最大值
    max_idx = np.argmax(frame_attn)
    ax.annotate(f"max@{key_indices[max_idx]}",
                xy=(key_indices[max_idx], frame_attn[max_idx]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=9, color="darkgreen", fontweight="bold")

plt.suptitle("Figure 4: Frame-wise Attention Weight Distribution\n"
             "(Query: frames 18-20 → Key: frames 0-20)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figure4_reproduction.svg", format="svg", bbox_inches="tight")
```

### 图表解读

- **X 轴**：Key Frame Index（0-20）
- **Y 轴**：Attention Logits（可以是负值）
- **柱状图**：每帧的平均注意力
- **折线图**：连接各帧，便于观察趋势

**Attention Sink 现象**：
- 如果 frame 0 的柱子明显高于其他帧 → 存在 sink
- 如果所有柱子高度接近 → 不存在明显 sink

---

## Cell 6: Per-Head Grid（每个 Head 的详细图）

### 代码

```python
for layer_idx in [0, 4]:
    if layer_idx not in layer_attention_data:
        continue

    frame_attn = layer_attention_data[layer_idx]["frame_attention"]
    key_indices = layer_attention_data[layer_idx]["key_frame_indices"]
    num_heads = frame_attn.shape[0]

    # 动态计算网格大小
    ncols = 4
    nrows = math.ceil(num_heads / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        head = frame_attn[h]

        # 计算 Sink Score: 首帧 - 中间帧平均
        first = head[0]
        middle_vals = head[1:-1]
        middle = middle_vals.mean() if len(middle_vals) > 0 else head.mean()
        sink_score = first - middle

        ax.bar(key_indices, head, alpha=0.7, width=0.8)
        ax.plot(key_indices, head, "o-", color="black", linewidth=1, markersize=2)
        ax.set_title(f"H{h+1} (Δ={sink_score:.2f})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的 subplot
    for k in range(num_heads, len(axes)):
        axes[k].axis("off")

    plt.suptitle(f"Layer {layer_idx+1}: Per-Head Attention Distribution\n"
                 f"Δ = First Frame - Middle Frames Average",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"figure4_layer{layer_idx+1}_per_head_grid.svg", format="svg")
```

### Sink Score 计算

```python
sink_score = first_frame_attn - middle_frames_avg

# Sink Score > 0: 首帧注意力高于平均 → 存在 Sink
# Sink Score ≈ 0: 首帧与中间帧相近 → 无明显 Sink
# Sink Score < 0: 首帧注意力低于平均 → 反向 Sink（罕见）
```

### 为什么用 Grid 而非单图

- 12 个 head 可能有不同模式
- 有些 head 是 "sink head"（高 Δ），有些是 "uniform head"（低 Δ）
- Grid 便于快速对比所有 head

---

## Cell 7: 热力图

### 代码

```python
for layer_idx in [0, 4]:
    if layer_idx not in layer_attention_data:
        continue

    frame_attn = layer_attention_data[layer_idx]["frame_attention"]
    key_indices = layer_attention_data[layer_idx]["key_frame_indices"]
    num_heads = frame_attn.shape[0]

    fig, ax = plt.subplots(figsize=(16, 6))

    # 绘制热力图
    im = ax.imshow(frame_attn, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Logits", fontsize=11)

    ax.set_xlabel("Key Frame Index", fontsize=12)
    ax.set_ylabel("Attention Head", fontsize=12)
    ax.set_title(f"Layer {layer_idx+1}: Attention Distribution Across All Heads",
                 fontsize=14, fontweight="bold")

    ax.set_yticks(range(num_heads))
    ax.set_yticklabels([f"H{i+1}" for i in range(num_heads)])
    ax.set_xticks(range(len(key_indices)))
    ax.set_xticklabels(key_indices)

    plt.tight_layout()
    plt.savefig(f"figure4_layer{layer_idx+1}_all_heads.svg", format="svg")
```

### 热力图解读

```
Y 轴: Attention Head (H1, H2, ..., H12)
X 轴: Key Frame Index (0, 1, 2, ..., 20)
颜色: Attention Logits 值（黄色高，紫色低）

Attention Sink 表现:
  - 第一列（frame 0）整体偏黄 → 所有 head 都对首帧高注意力
  - 或某几行的第一列特别黄 → 部分 head 是 sink head
```

---

## Cell 8: 跨层对比

### 代码

```python
fig, ax = plt.subplots(figsize=(14, 5))
colors_cmp = plt.cm.viridis(np.linspace(0.2, 0.8, len(layer_attention_data)))

for idx, layer_idx in enumerate(sorted(layer_attention_data.keys())):
    frame_attn = layer_attention_data[layer_idx]["frame_attention"]
    key_indices = layer_attention_data[layer_idx]["key_frame_indices"]

    # 计算所有 head 的平均和标准差
    mean_attn = frame_attn.mean(axis=0)
    std_attn = frame_attn.std(axis=0)

    # 绘制均值线
    ax.plot(key_indices, mean_attn, "o-", color=colors_cmp[idx],
            linewidth=2, markersize=5, label=f"Layer {layer_idx+1}", alpha=0.9)

    # 绘制标准差带
    ax.fill_between(key_indices,
                    mean_attn - std_attn,
                    mean_attn + std_attn,
                    color=colors_cmp[idx], alpha=0.2)

ax.set_xlabel("Key Frame Index", fontsize=12)
ax.set_ylabel("Mean Attention Weight (Logits)", fontsize=12)
ax.set_title("Cross-Layer Head-Averaged Attention Distribution",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("figure4_layer_comparison.svg", format="svg")
```

### 为什么需要跨层对比

- **浅层（Layer 1）**：可能更关注局部特征，sink 现象可能不明显
- **深层（Layer 5）**：可能更关注全局语义，sink 现象可能更强
- **标准差带**：显示 head 间的一致性（窄带 = head 间一致）

---

## Cell 9: 统计摘要

### 代码

```python
print("=" * 70)
print("统计摘要")
print("=" * 70)

for layer_idx in sorted(layer_attention_data.keys()):
    frame_attn = layer_attention_data[layer_idx]["frame_attention"]
    mean_attn = frame_attn.mean(axis=0)

    first = mean_attn[0]
    middle = mean_attn[1:-1].mean()
    last = mean_attn[-1]

    print(f"Layer {layer_idx+1}:")
    print(f"  首帧 (frame 0): {first:.4f}")
    print(f"  中间帧 (frames 1-19 avg): {middle:.4f}")
    print(f"  末帧 (frame 20): {last:.4f}")
    print(f"  首帧/中间 比值: {first/middle:.2f}x")
    print(f"  首帧 - 末帧: {first - last:.4f}")
```

### 输出示例

```
======================================================================
统计摘要
======================================================================
Layer 1:
  首帧 (frame 0): 2.9479
  中间帧 (frames 1-19 avg): 3.6364
  末帧 (frame 20): 3.2246
  首帧/中间 比值: 0.81x
  首帧 - 末帧: -0.2767
Layer 5:
  首帧 (frame 0): 2.0843
  中间帧 (frames 1-19 avg): 2.5531
  末帧 (frame 20): 2.5398
  首帧/中间 比值: 0.82x
  首帧 - 末帧: -0.4555
```

### 关键指标解读

| 指标 | 含义 | Attention Sink 的预期值 |
|------|------|------------------------|
| 首帧/中间 比值 | 首帧相对强度 | > 1.0（如 1.5x, 2x） |
| 首帧 - 末帧 | 时序偏好 | > 0（偏好早期帧） |

**当前结果解读**：
- 比值 0.81x < 1.0 → 首帧注意力反而**低于**中间帧
- 这可能说明基础 Wan 模型**没有**明显的 attention sink

---

## 完整工作流

```bash
# 1. 提取注意力权重
PYTHONPATH=. python experiments/run_extraction_figure4.py \
    --no_checkpoint \
    --output_path cache/attention_cache_wan_base.pt

# 2. 运行 notebook 生成图表
cd notebooks
jupyter lab plt_fig4.ipynb

# 3. 查看生成的 SVG 文件
ls -la *.svg
```

---

## 常见问题

### Q1: 为什么 Y 轴是 "Logits" 而非 "Probability"？

Logits（pre-softmax）保留了原始注意力强度差异。Softmax 后的概率会被归一化，压缩差异。论文 Figure 4 的 Y 轴范围 [-4, 6] 也说明使用的是 logits。

### Q2: 如何判断是否存在 Attention Sink？

- **首帧/中间 比值 > 1.2**：明显 sink
- **首帧/中间 比值 ≈ 1.0**：无 sink
- **某些 head 的 Δ 值 > 2.0**：该 head 是 sink head

### Q3: 基础模型没有 sink 是否正常？

根据论文，sink 现象在**训练过程中**逐渐形成。基础 Wan 模型可能没有经过足够长的 causal attention 训练来产生 sink。Self-Forcing 训练后的模型可能会表现出更强的 sink。
