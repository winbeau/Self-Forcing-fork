# bench_infer.sh — Self-Forcing Batch Inference

Multi-GPU 批量推理脚本，适用于对比实验。从 prompt 文本文件生成视频，输出标准化命名 (`video_000.mp4`, `video_001.mp4`, ...) 并附带 `prompts.csv` 索引。

---

## Experiment Setup

### 评测基准

| 项目 | 值 |
|---|---|
| **Benchmark** | MovieGenVideoBench (32 prompts) |
| **Prompt 文件** | `prompts/MovieGenVideoBench_num32.txt` |
| **Seed** | `0` (多卡时 GPU *i* 种子为 `seed + i`) |

### 生成配置

| 项目 | 标准 (5s) | 长视频 (30s) |
|---|---|---|
| **Config** | `self_forcing_dmd.yaml` | `self_forcing_dmd_long.yaml` |
| **像素分辨率** | 480 × 832 | 480 × 832 |
| **Latent 帧数** | 21 | 120 |
| **像素帧数** | 81 | 477 |
| **FPS** | 16 | 16 |
| **视频时长** | ~5s | ~30s |
| **去噪步数** | 4 (`[1000, 750, 500, 250]`) | 4 (`[1000, 750, 500, 250]`) |
| **Guidance scale** | 3.0 | 3.0 |
| **Timestep shift** | 5.0 | 5.0 |
| **Warp denoising** | Yes | Yes |
| **EMA 权重** | Yes (`--use_ema`) | Yes (`--use_ema`) |
| **KV cache** | 固定 (21帧) | Rolling (`local_attn_size=21, sink_size=1`) |

> **帧数换算：** `像素帧 = (latent_frames - 1) × 4 + 1`

### 因果生成 Block 结构

| 项目 | 标准 (21帧) | 长视频 (120帧) |
|---|---|---|
| **每 block 帧数** | 3 latent 帧 | 3 latent 帧 |
| **Block 数量** | 7 | 40 |
| **Block 序列** | `[3] × 7` | `[3] × 40` |

### 模型变体

| 项目 | DMD 14B | SID 14B | SID 1.3B |
|---|---|---|---|
| **Config** | `self_forcing_dmd.yaml` | `self_forcing_sid.yaml` | `self_forcing_sid.yaml` |
| **real_name** | `Wan2.1-T2V-14B` | `Wan2.1-T2V-14B` | `Wan2.1-T2V-1.3B` |
| **Distribution loss** | DMD | SiD | SiD |
| **Hidden dim** | 5120 | 5120 | 1536 |
| **FFN dim** | 13824 | 13824 | 8960 |
| **Attention heads** | 40 | 40 | 12 |
| **Transformer layers** | 40 | 40 | 30 |
| **Precision** | bfloat16 | bfloat16 | bfloat16 |

### Latent 空间

| 项目 | 值 |
|---|---|
| **Latent shape** | `[B, T, 16, 60, 104]` — T=21 (标准) 或 T=120 (长视频) |
| **Latent 通道数** | 16 |
| **VAE stride** | `(4, 8, 8)` — (temporal, height, width) |
| **VAE 空间下采样** | 8× (480/8=60, 832/8=104) |
| **VAE 时间下采样** | 4× |

### 文本编码器

| 项目 | 值 |
|---|---|
| **模型** | UMT5-XXL (encoder-only) |
| **Hidden dim** | 4096 |
| **Max token length** | 512 |
| **Precision** | bfloat16 |

---

## 复现命令

### 标准实验 (~5s 视频)

```bash
# DMD 14B (推荐)
bash scripts/bench_infer.sh \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --num_gpus 4 \
  --output outputs/dmd_14B \
  --use_ema

# SID 1.3B
bash scripts/bench_infer.sh \
  --config configs/self_forcing_sid.yaml \
  --checkpoint checkpoints/self_forcing_sid.pt \
  --num_gpus 4 \
  --output outputs/sid_1_3B \
  --use_ema
```

### 长视频实验 (~30s 视频, Rolling KV Cache)

```bash
# 120 latent 帧
bash scripts/bench_infer.sh \
  --config configs/self_forcing_dmd_long.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --num_frames 120 \
  --num_gpus 4 \
  --output outputs/long_120f \
  --use_ema

# 先用 42 帧做 smoke test
bash scripts/bench_infer.sh \
  --config configs/self_forcing_dmd_long.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --num_frames 42 \
  --output outputs/long_test \
  --use_ema
```

**Rolling KV Cache 原理：**
- `local_attn_size=21`：滑动窗口大小 = 训练上下文长度（21 帧），最小化分布偏移
- `sink_size=1`：保留第一帧作为全局锚点，维持长视频一致性
- KV cache 自动滚动淘汰旧 token，VRAM 占用恒定

> **注意：** 120 帧 VAE 解码可能 OOM，如遇到需在 `causal_inference.py` 中添加分块解码。

---

## CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/self_forcing_dmd.yaml` | 模型配置文件 |
| `--checkpoint` | `checkpoints/self_forcing_dmd.pt` | 权重路径 (`.pt`) |
| `--data` | `prompts/MovieGenVideoBench_num32.txt` | Prompt 文件，每行一条 |
| `--output` | `outputs/movie_gen_bench` | 输出目录 |
| `--num_gpus` | 自动检测全部 GPU | 并行 GPU 数量 |
| `--num_frames` | `21` | 生成的 **latent 帧数** (非像素帧数) |
| `--seed` | `0` | 随机种子 (多卡时 GPU *i* 的种子为 `seed + i`) |
| `--use_ema` | `true` | 使用 EMA 权重推理 |

---

## 输出结构

```
outputs/movie_gen_bench/
├── prompts.csv          # index ↔ prompt 对应表
├── video_000.mp4        # 第 0 条 prompt 的视频
├── video_001.mp4
├── ...
└── video_031.mp4        # 共 32 条
```

**prompts.csv 格式：**

```csv
index,prompt
000,"A stylish woman strolls down a bustling Tokyo street, ..."
001,"A stunning mid-afternoon landscape photograph ..."
```

不同实验的 `prompts.csv` 内容一致，视频序号一一对应，可直接做逐条对比。

---

## 多卡并行机制

- 使用 `torchrun --nproc_per_node=N` 启动，NCCL 后端
- `DistributedSampler(shuffle=False, drop_last=True)` 将 prompt 均匀分配给各 GPU
- 每张 GPU 独立保存自己负责的视频，无跨卡聚合
- **注意：** `drop_last=True` 要求 prompt 数量能被 GPU 数整除，否则末尾 prompt 会被丢弃
  - 32 prompts: 1/2/4/8 卡均可整除
- 每张 GPU 的种子为 `seed + local_rank`，保证不同卡生成不同噪声

## 文件重命名策略

`inference.py` 原始输出格式为 `{idx}-{seed_idx}_{regular|ema}.mp4`（如 `5-0_regular.mp4`）。

脚本启动后台 watcher 进程，每 2 秒扫描输出目录，检测到新文件后：
1. 等待文件写入完成（1 秒内文件大小不变）
2. 立即重命名为 `video_{idx:03d}.mp4`

推理结束后再做一次最终扫描，确保无遗漏。
