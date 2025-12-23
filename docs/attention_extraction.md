# 全注意力矩阵提取逻辑

本文档描述 `experiments/run_extraction_each.py` 的注意力提取逻辑。

## 概述

提取单层 Transformer 的完整 frame×frame 注意力矩阵，用于分析 Self-Forcing 模型的 attention sink 现象。

## 核心概念

### 1. Block 结构

Self-Forcing 模型使用 block-wise autoregressive 生成：

```
num_frames = 21
num_frame_per_block = 3
independent_first_frame = True

block_sizes = [1, 3, 3, 3, 3, 3, 3]  # 总共 7 个 block，19 帧
```

- Block 0: 第 1 帧（独立）
- Block 1-6: 每个 3 帧

### 2. 注意力调用索引

每个 Transformer block 有 2 次 attention 调用：

| 调用类型 | 索引规则 | K 长度 |
|---------|---------|--------|
| Self-Attention | 偶数 (0, 2, 4, ...) | 视频 tokens (32760) |
| Cross-Attention | 奇数 (1, 3, 5, ...) | 文本 tokens (512) |

**索引映射**：
```python
layer_index = 3  # 第 3 层
self_attn_idx = 2 * layer_index  # = 6
```

### 3. Block-based Causality

与传统的 frame-level causality (k ≤ q) 不同，Self-Forcing 使用 **block-based causality**：

- Block 内所有 Q frames 可以 attend 到该 block 结束为止的所有 K frames
- 例：Block 1 的 Q frames 1-3 都可以看到 K frames 0-3

```
Block 0: Q=[0],     K=[0]        (1 帧)
Block 1: Q=[1,2,3], K=[0,1,2,3]  (4 帧)
Block 2: Q=[4,5,6], K=[0..6]     (7 帧)
...
Block 6: Q=[16,17,18], K=[0..18] (19 帧)
```

## 提取流程

### Step 1: 启用注意力捕获

```python
from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE

ATTENTION_WEIGHT_CAPTURE.enable(
    layer_indices=[self_attn_idx],  # 要捕获的层索引
    capture_logits=True,             # 捕获 pre-softmax logits
    num_layers=num_layers * 2        # 总调用次数 = 层数 × 2
)
```

### Step 2: 运行推理

```python
output = pipeline.inference(
    noise=noise,
    text_prompts=[prompt],
    return_latents=True,
)
captured_weights = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
```

### Step 3: 构建完整矩阵

按 K 长度排序捕获的注意力，然后逐 block 组装：

```python
# 初始化 [num_heads, num_frames, num_frames]
full_frame_attn = torch.zeros(num_heads, num_frames, num_frames)

current_q_start = 0
for block_idx, block_size in enumerate(block_sizes):
    expected_k_frames = sum(block_sizes[:block_idx + 1])

    # 找到匹配的 attention (按 K 长度)
    matching_attn = find_matching_attention(expected_k_frames)

    # 提取 token-level attention
    attn_logits = matching_attn['attn_weights'][0]  # [num_heads, Lq, Lk]

    # 聚合到 frame-level
    for h in range(num_heads):
        for qf_local in range(q_frames_in_block):
            qf_global = current_q_start + qf_local
            for kf in range(k_frames_total):
                # 平均 token pair 的注意力
                frame_attn = attn_logits[h, q_tokens, k_tokens].mean()
                full_frame_attn[h, qf_global, kf] = frame_attn

    current_q_start += block_size
```

### Step 4: 保存数据

```python
save_data = {
    'layer_index': layer_index,
    'full_frame_attention': full_frame_attn,      # [num_heads, Q, K]
    'last_block_frame_attention': last_block_attn, # [num_heads, K]
    'is_logits': True,
    'prompt': prompt,
    'num_frames': num_frames,
    'frame_seq_length': 1560,  # tokens per frame
    'num_heads': 12,
    'block_sizes': block_sizes,
    'last_block_query_frames': [16, 17, 18],
}
torch.save(save_data, output_path)
```

## Token-to-Frame 聚合

每帧包含 1560 个 tokens (60×104/4)：

```python
frame_seq_length = 1560

# 对于 Q frame qf 和 K frame kf
q_start = qf * frame_seq_length
q_end = (qf + 1) * frame_seq_length
k_start = kf * frame_seq_length
k_end = (kf + 1) * frame_seq_length

# 取所有 token pair 的平均
frame_attention = attn_logits[h, q_start:q_end, k_start:k_end].mean()
```

## 使用方法

```bash
# 提取第 3 层
PYTHONPATH=. python experiments/run_extraction_each.py \
    --layer_index 3 \
    --output_path cache/layer3.pt \
    --checkpoint_path checkpoints/self_forcing_dmd.pt

# 使用原始 Wan 模型（不加载 checkpoint）
PYTHONPATH=. python experiments/run_extraction_each.py \
    --layer_index 3 \
    --output_path cache/layer3_wan.pt \
    --no_checkpoint
```

## 输出数据结构

| 字段 | 形状 | 说明 |
|-----|------|------|
| `full_frame_attention` | [12, 19, 19] | 完整 frame×frame 注意力矩阵 |
| `last_block_frame_attention` | [12, 19] | 最后 block 对各帧的平均注意力 |
| `block_sizes` | [7] | 每个 block 的帧数 |
| `num_heads` | int | 注意力头数 (12) |
| `frame_seq_length` | int | 每帧 token 数 (1560) |

## 相关文件

- `experiments/run_extraction_each.py` - 提取脚本
- `wan/modules/attention.py` - ATTENTION_WEIGHT_CAPTURE 机制
- `tests/test_extraction_logic.py` - 单元测试
