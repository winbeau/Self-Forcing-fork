# bench_infer.sh — Self-Forcing Batch Inference

Multi-GPU 批量推理脚本，适用于对比实验。从 prompt 文本文件生成视频，输出标准化命名 (`video_000.mp4`, `video_001.mp4`, ...) 并附带 `prompts.csv` 索引。

---

## Quick Start

```bash
bash scripts/bench_infer.sh --checkpoint /path/to/self_forcing_dmd.pt

# 指定 GPU 数量和输出目录
bash scripts/bench_infer.sh \
  --checkpoint /path/to/self_forcing_dmd.pt \
  --num_gpus 4 \
  --output outputs/exp_dmd_14B
```

---

## CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/self_forcing_dmd.yaml` | 模型配置文件 |
| `--checkpoint` | *(必填)* | 训练好的权重路径 (`.pt`) |
| `--data` | `prompts/MovieGenVideoBench_num32.txt` | Prompt 文件，每行一条 |
| `--output` | `outputs/movie_gen_bench` | 输出目录 |
| `--num_gpus` | 自动检测全部 GPU | 并行 GPU 数量 |
| `--num_frames` | `21` | 生成的 **latent 帧数** (非像素帧数) |
| `--seed` | `0` | 随机种子 (多卡时 GPU *i* 的种子为 `seed + i`) |
| `--use_ema` | `false` | 使用 EMA 权重推理 |

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

---

## 视频生成规格 (对比实验关键参数)

### 分辨率与帧数

| 项目 | 值 | 来源 |
|---|---|---|
| **像素分辨率** | **480 × 832** (H×W) | `default_config.yaml` → `height`, `width` |
| **输出总像素帧数** | **81 帧** | `(latent_frames - 1) × temporal_stride + 1 = (21-1)×4+1` |
| **Latent 帧数** | 21 帧 | `--num_output_frames 21` |
| **输出 FPS** | **16 fps** | `inference.py` → `write_video(..., fps=16)` |
| **视频时长** | **≈ 5.06 秒** | `81 / 16 = 5.0625s` |
| **输出格式** | MP4 (H.264) | `torchvision.io.write_video` |

### Latent 空间

| 项目 | 值 |
|---|---|
| **Latent shape** | `[batch, 21, 16, 60, 104]` — (B, T, C, H, W) |
| **Latent 通道数** | 16 |
| **VAE 空间下采样** | 8× (480/8=60, 832/8=104) |
| **VAE 时间下采样** | 4× |
| **VAE stride** | `(4, 8, 8)` — (temporal, height, width) |

### 因果生成 Block 结构

| 项目 | 值 |
|---|---|
| **每 block 帧数** | 3 latent 帧 (`num_frame_per_block: 3`) |
| **Block 数量** | 7 blocks (`21 / 3 = 7`) |
| **independent_first_frame** | `false` |
| **Block 序列** | `[3, 3, 3, 3, 3, 3, 3]` |

### 去噪 (Denoising)

| 项目 | DMD config | SID config |
|---|---|---|
| **去噪步数** | 4 步 | 4 步 |
| **denoising_step_list** | `[1000, 750, 500, 250]` | `[1000, 750, 500, 250]` |
| **warp_denoising_step** | `true` | `true` |
| **Guidance scale** | 3.0 | 3.0 |
| **Timestep shift** | 5.0 | 5.0 |
| **num_train_timesteps** | 1000 | 1000 |

### 模型架构

| 项目 | Wan2.1-T2V-14B | Wan2.1-T2V-1.3B |
|---|---|---|
| **Hidden dim** | 5120 | 1536 |
| **FFN dim** | 13824 | 8960 |
| **Attention heads** | 40 | 12 |
| **Transformer layers** | 40 | 30 |
| **Patch size** | (1, 2, 2) | (1, 2, 2) |
| **Freq dim** | 256 | 256 |
| **QK norm** | Yes | Yes |
| **Precision** | bfloat16 | bfloat16 |
| **Config 名** | `self_forcing_dmd.yaml` | `self_forcing_sid.yaml` |

### 文本编码器

| 项目 | 值 |
|---|---|
| **模型** | UMT5-XXL (encoder-only) |
| **Hidden dim** | 4096 |
| **Attention heads** | 64 |
| **Encoder layers** | 24 |
| **Max token length** | 512 |
| **Tokenizer** | `google/umt5-xxl` |
| **Precision** | bfloat16 |

### VAE

| 项目 | 值 |
|---|---|
| **类型** | WanVAE (3D causal VAE) |
| **Checkpoint** | `Wan2.1_VAE.pth` |
| **Stride** | `(4, 8, 8)` |
| **z_dim** | 16 |
| **基础维度** | 128 |
| **维度乘子** | `[1, 2, 4, 4]` |
| **残差块数** | 2 per stage |

### Negative Prompt (固定)

```
色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，
最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，
画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，
杂乱的背景，三条腿，背景人很多，倒着走
```

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

---

## 对比实验示例

```bash
# DMD 14B
bash scripts/bench_infer.sh \
  --config configs/self_forcing_dmd.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --output outputs/dmd_14B \
  --num_gpus 4

# SID 1.3B
bash scripts/bench_infer.sh \
  --config configs/self_forcing_sid.yaml \
  --checkpoint checkpoints/self_forcing_sid.pt \
  --output outputs/sid_1_3B \
  --num_gpus 4

# EMA 权重
bash scripts/bench_infer.sh \
  --config configs/self_forcing_dmd.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --output outputs/dmd_14B_ema \
  --use_ema
```

不同实验的 `prompts.csv` 内容一致，视频序号一一对应，可直接做逐条对比。

---

## 长视频生成 (Rolling KV Cache)

使用 `self_forcing_dmd_long.yaml` 配置启用 rolling KV cache，突破默认 21 帧限制，生成更长视频。

**原理：**
- `local_attn_size=21`：滑动窗口大小与训练上下文长度一致（21 帧），最小化分布偏移
- `sink_size=1`：保留第一帧作为全局锚点，维持长视频一致性
- KV cache 自动滚动淘汰旧 token，VRAM 占用恒定

**帧数换算：** `像素帧 = (latent_frames - 1) × 4 + 1`

| Latent 帧数 | 像素帧数 | 视频时长 (16fps) |
|---|---|---|
| 21 (默认) | 81 | ~5s |
| 42 | 165 | ~10s |
| 60 | 237 | ~15s |
| 120 | 477 | ~30s |

```bash
# 120 latent 帧 (~30s 视频)
bash scripts/bench_infer.sh \
  --config configs/self_forcing_dmd_long.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --num_frames 120 \
  --num_gpus 4 \
  --output outputs/long_120f \
  --use_ema

# 先用较短帧数测试
bash scripts/bench_infer.sh \
  --config configs/self_forcing_dmd_long.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --num_frames 42 \
  --output outputs/long_test \
  --use_ema
```

**注意：** 120 帧 VAE 解码可能 OOM，如遇到需在 `causal_inference.py` 中添加分块解码。
