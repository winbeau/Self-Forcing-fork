# run_extraction_figure4.py 详解

## 概述

本脚本用于从 Wan2.1 视频生成模型中提取注意力权重，用于复现 Deep Forcing 论文的 Figure 4（Attention Sink 现象分析）。

**核心思路**：直接使用模型原生的 `ATTENTION_WEIGHT_CAPTURE` 机制，在真实推理过程中捕获 attention，确保与模型实际计算完全一致。

## 文件位置

```
experiments/run_extraction_figure4.py
```

## 依赖关系

```
run_extraction_figure4.py
    ├── pipeline/causal_inference.py      # 推理 Pipeline
    ├── wan/modules/attention.py          # ATTENTION_WEIGHT_CAPTURE 机制
    ├── utils/misc.py                     # set_seed 等工具
    └── configs/self_forcing_dmd.yaml     # 配置文件
```

---

## 核心逻辑流程

```
1. 加载配置和模型
       ↓
2. 创建输入噪声 [B, T, C, H, W]
       ↓
3. 启用 ATTENTION_WEIGHT_CAPTURE
       ↓
4. 执行推理（自动捕获 attention）
       ↓
5. 按层分组，选择最后一个 block 的 attention
       ↓
6. 保存数据到 .pt 文件
```

---

## 关键代码详解

### 1. 命令行参数解析

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="cache/attention_cache_figure4.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--layer_indices", type=int, nargs='+', default=[0, 4])
    parser.add_argument("--no_checkpoint", action="store_true", default=False,
                        help="不加载 checkpoint，使用原始 Wan2.1 基础模型")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--layer_indices` | 要捕获的层索引，默认 `[0, 4]` 对应 Layer 1 和 Layer 5 |
| `--no_checkpoint` | **重要**：使用基础 Wan 模型而非训练后的模型 |
| `--num_frames` | 生成帧数，默认 21 帧 |

### 2. 模型加载逻辑

```python
# 加载配置（合并默认配置）
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# 初始化 Pipeline
pipeline = CausalInferencePipeline(args=config, device=device)

# Checkpoint 加载逻辑
if args.no_checkpoint:
    print("使用原始 Wan2.1 基础模型（不加载任何 checkpoint）")
elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    key = 'generator_ema' if args.use_ema else 'generator'
    pipeline.generator.load_state_dict(state_dict[key])
```

**为什么需要 `--no_checkpoint`**：
- Deep Forcing 论文分析的是**基础模型**的 attention sink 现象
- Self-Forcing 训练后的模型可能已经改变了 attention 分布
- 对比两者可以验证训练效果

### 3. 启用注意力捕获（核心）

```python
# 获取模型层数
num_layers = len(pipeline.generator.model.blocks)  # Wan 模型有 30 层

# 启用 attention 捕获
ATTENTION_WEIGHT_CAPTURE.enable(
    layer_indices=args.layer_indices,  # 只捕获指定层
    capture_logits=True,               # 捕获 pre-softmax logits（不是概率）
    num_layers=num_layers              # 用于模块化索引
)
```

**参数详解**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `layer_indices` | `[0, 4]` | 捕获第 1 层和第 5 层 |
| `capture_logits` | `True` | 返回 QK^T / √d（论文 Y 轴范围 [-4, 6]） |
| `num_layers` | `30` | 用于 `current_layer_idx % 30` 计算有效层索引 |

**为什么要用 `capture_logits=True`**：
```
Softmax 前（logits）: 可以是负值，范围如 [-4, 6]，保留原始注意力强度差异
Softmax 后（probs）:  归一化到 [0, 1]，压缩了差异，不利于分析
```

### 4. 执行推理

```python
# 创建输入噪声
noise = torch.randn(
    [batch_size, args.num_frames, 16, 60, 104],  # [B, T, C, H, W]
    device=device,
    dtype=torch.bfloat16
)

try:
    output = pipeline.inference(
        noise=noise,
        text_prompts=[args.prompt],
        return_latents=True,
    )
finally:
    captured_weights = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
    ATTENTION_WEIGHT_CAPTURE.disable()
```

**噪声张量维度**：
- `B=1`: batch size
- `T=21`: 帧数
- `C=16`: latent 通道数
- `H=60, W=104`: latent 空间分辨率（原始 480×832 / 8）

### 5. 选择最后一个 Block 的 Attention（关键）

```python
# 按层分组
layer_attentions = {}
for attn in captured_weights:
    layer_idx = attn['layer_idx']
    if layer_idx not in layer_attentions:
        layer_attentions[layer_idx] = []
    layer_attentions[layer_idx].append(attn)

# 每层选择 K 长度最大的（对应最后一个 temporal block）
final_attentions = []
for layer_idx in sorted(layer_attentions.keys()):
    attns = layer_attentions[layer_idx]
    # 按 K 的长度排序，取最大的
    attns_sorted = sorted(attns, key=lambda x: x['k_shape'][1], reverse=True)
    selected = attns_sorted[0]
    final_attentions.append(selected)
```

**为什么选择 K 最大的**：

视频生成是 block-wise autoregressive：
```
Block 1: 生成 frames 0-2,   K 包含 frames 0-2   (K_len = 3 * 1560)
Block 2: 生成 frames 3-5,   K 包含 frames 0-5   (K_len = 6 * 1560)
...
Block 7: 生成 frames 18-20, K 包含 frames 0-20  (K_len = 21 * 1560) ← 最大
```

最后一个 block 的 K 包含所有历史帧，这正是 Figure 4 需要分析的。

### 6. 推断帧参数

```python
# 从捕获数据推断参数
first_attn = final_attentions[0]
k_len = first_attn['k_shape'][1]  # K 的 token 总数
q_len = first_attn['q_shape'][1]  # Q 的 token 总数

# 推断 frame_seq_length（每帧的 token 数）
# Q 有 num_frame_per_block 帧（通常 3 帧）
frame_seq_length = q_len // num_frame_per_block  # 4680 / 3 = 1560

# 推断 K 帧数
num_key_frames = k_len // frame_seq_length  # 32760 / 1560 = 21
```

**frame_seq_length 计算**：
```
原始分辨率: 480 × 832
Latent 分辨率: 60 × 104
Patch 大小: 2 × 2
Token 数/帧: (60/2) × (104/2) = 30 × 52 = 1560
```

### 7. 保存数据结构

```python
save_data = {
    'attention_weights': [],           # 注意力权重列表
    'prompt': args.prompt,             # 使用的 prompt
    'num_frames': args.num_frames,     # 总帧数
    'frame_seq_length': frame_seq_length,  # 每帧 token 数
    'num_frame_per_block': num_frame_per_block,  # 每 block 帧数
    'layer_indices': args.layer_indices,  # 捕获的层
    'is_logits': True,                 # 是否为 logits
    'capture_method': 'native_inference_capture',
    'query_frames': list(range(args.num_frames - num_frame_per_block, args.num_frames)),
    'key_frames': list(range(num_key_frames)),
}

for attn in final_attentions:
    save_data['attention_weights'].append({
        'layer_idx': attn['layer_idx'],
        'attn_logits': attn['attn_weights'].to(torch.float16),  # [B, N, Q, K]
        'is_logits': attn['is_logits'],
        'num_frames': args.num_frames,
        'frame_seq_length': frame_seq_length,
        'num_heads': attn['attn_weights'].shape[1],
        'num_query_frames': num_frame_per_block,
        'num_key_frames': num_key_frames,
    })
```

**attention 张量维度**：
```
attn_logits: [B, N, Q, K]
  B = 1          (batch size)
  N = 12         (attention heads)
  Q = 4680       (3 帧 × 1560 tokens/帧)
  K = 32760      (21 帧 × 1560 tokens/帧)
```

---

## 输出示例

```
Using device: cuda:0
GPU: NVIDIA A100-SXM4-80GB
Initializing inference pipeline...
使用原始 Wan2.1 基础模型（不加载任何 checkpoint）

Num frames: 21
Num frames per block: 3
Layer indices to capture: [0, 4]
Prompt: A majestic eagle soaring through a cloudy sky, cinematic lighting

============================================================
在推理过程中捕获 attention（使用模型原生机制）
============================================================
模型层数: 30

执行推理...
Output latent shape: torch.Size([1, 21, 16, 60, 104])

捕获到 56 个 attention 矩阵

Layer 0: selected attention with Q=torch.Size([1, 4680, 12, 128]), K=torch.Size([1, 32760, 12, 128])
Layer 4: selected attention with Q=torch.Size([1, 4680, 12, 128]), K=torch.Size([1, 32760, 12, 128])

推断的参数:
  frame_seq_length: 1560
  num_key_frames: 21
  num_query_frames: 3

Saved to: cache/attention_cache_wan_base.pt
```

---

## 使用示例

```bash
# 基础模型（复现 Figure 4）
PYTHONPATH=. python experiments/run_extraction_figure4.py \
    --no_checkpoint \
    --output_path cache/attention_cache_wan_base.pt

# Self-Forcing 模型（对比分析）
PYTHONPATH=. python experiments/run_extraction_figure4.py \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_path cache/attention_cache_self_forcing.pt

# 自定义参数
PYTHONPATH=. python experiments/run_extraction_figure4.py \
    --layer_indices 0 4 8 12 \
    --num_frames 33 \
    --prompt "A cat walking on the beach" \
    --seed 123
```

---

## 常见问题

### Q1: 为什么捕获到 56 个而不是 14 个 attention？

推理过程有 7 个 block × 4 个 denoising step = 28 次前向传播。
每次前向传播中，layer 0 和 layer 4 各被调用一次，共 28 × 2 = 56 次。

### Q2: 为什么要用模块化索引？

Wan 模型有 30 层。推理时 `current_layer_idx` 从 0 递增到 839（30 × 28 - 1）。
使用 `% 30` 才能正确识别每次经过 layer 0 和 layer 4。

### Q3: 输出的 attention shape 为什么是 [B, N, Q, K] 而不是 [B, Q, N, K]？

这是 PyTorch attention 的标准格式。在 `attention_with_weights` 函数中，
Q/K/V 先 transpose 为 `[B, N, L, C]`，计算 `QK^T` 后得到 `[B, N, Lq, Lk]`。
