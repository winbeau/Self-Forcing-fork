# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self Forcing is a research codebase for training autoregressive video diffusion models. It resolves train-test distribution mismatch by **simulating the inference process during training** with autoregressive rollout and KV caching, enabling real-time streaming video generation.

Built on top of [CausVid](https://github.com/tianweiy/CausVid) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). The `wan/` directory contains modified code from Wan2.1 (Apache-2.0 License).

## Installation & Setup

```bash
# Install dependencies with uv
uv sync
uv pip install flash-attn --no-build-isolation
```

## Common Commands

### Download Checkpoints

```bash
# Download base Wan model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B

# Download self-forcing checkpoint
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .

# Download for training
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```

### Inference

```bash
# GUI demo (recommended for testing)
python demo.py

# CLI inference - DMD checkpoint
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema

# CLI inference - SID checkpoint
python inference.py \
    --config_path configs/self_forcing_sid.yaml \
    --output_folder videos/self_forcing_sid \
    --checkpoint_path checkpoints/self_forcing_sid.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema

# Distributed inference (multi-GPU)
torchrun --nproc_per_node=8 inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

### Training

```bash
# Self Forcing Training with DMD
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb

# Training with SID
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_sid.yaml \
  --logdir logs/self_forcing_sid \
  --disable-wandb
```

## Architecture

### Core Components

**Training Pipeline (`pipeline/self_forcing_training.py`)**
- `SelfForcingTrainingPipeline`: Implements autoregressive rollout with KV caching during training
- Simulates inference process by generating video in blocks (chunk-wise autoregressive)
- Maintains KV cache across blocks to match inference behavior
- Key parameters: `num_frame_per_block` (frames per generation block, typically 3)

**Inference Pipelines**
- `CausalInferencePipeline` (pipeline/causal_inference.py): Few-step inference with DMD/SID checkpoints
- `CausalDiffusionInferencePipeline` (pipeline/causal_diffusion_inference.py): Multi-step diffusion inference
- Both use KV caching for efficient autoregressive generation

**Models (`model/`)**
- `BaseModel` (base.py): Base class initializing generator, text encoder, VAE, and score models
- `CausalDiffusion` (diffusion.py): Main training model with self-forcing logic
- `DMD` (dmd.py): Distribution Matching Distillation variant
- `SID` (sid.py): Score Identity Distillation variant
- `GANTrainer` (gan.py): GAN-based training (requires video data)
- `ODERegression` (ode_regression.py): ODE initialization

**Trainers (`trainer/`)**
- `DiffusionTrainer`: Standard diffusion training
- `ScoreDistillationTrainer`: Training with DMD/SID (used for self-forcing)
- `GANTrainer`: GAN-based training
- `ODETrainer`: ODE initialization training

**Wan Wrapper (`utils/wan_wrapper.py`)**
- `WanDiffusionWrapper`: Wrapper around Wan2.1 transformer model
- `WanTextEncoder`: Text encoder (T5/CLIP)
- `WanVAEWrapper`: VAE for encoding/decoding latents
- Critical abstraction layer - all model interactions go through these wrappers

### Key Architectural Patterns

**Causal vs Non-Causal Models**
- Causal models (`is_causal=True`): Use KV caching for autoregressive generation
- Non-causal models: Standard bidirectional attention
- The generator is causal during training/inference; score models can be non-causal

**Block-wise Generation**
- Videos generated in blocks of `num_frame_per_block` frames (typically 3)
- KV cache maintained across blocks for efficient generation
- First block may skip initial frames (e.g., frames 0-2) during decoding

**Training vs Inference Gap**
- Self-forcing bridges this gap by using autoregressive rollout during training
- Training simulates inference: generates blocks sequentially with KV cache
- Unlike teacher forcing, this prevents distribution mismatch

**Denoising Steps**
- Few-step models use `denoising_step_list` (e.g., [1000, 750, 500, 250])
- `warp_denoising_step`: Maps step indices to actual timesteps from scheduler
- Multi-step models use full diffusion process

### Configuration System

All configs inherit from `configs/default_config.yaml` and are merged with OmegaConf:
```python
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
```

Key config parameters:
- `trainer`: Type of trainer (diffusion, score_distillation, gan, ode)
- `distribution_loss`: Loss type (dmd, sid)
- `denoising_step_list`: Timesteps for few-step inference
- `num_frame_per_block`: Frames per generation block
- `causal`: Enable causal (autoregressive) generation
- `independent_first_frame`: Special handling for first frame
- `image_or_video_shape`: [batch, frames, channels, height, width] - [1, 21, 16, 60, 104] is standard

### Memory Management (`demo_utils/memory.py`)

- `DynamicSwapInstaller`: Swaps models between CPU/GPU for low-memory setups
- `move_model_to_device_with_memory_preservation`: Moves models while preserving memory
- Low memory mode activates when VRAM < 40GB
- Demo supports TAEHV-VAE (faster, lower quality) and FP8 quantization for speedup

## Important Notes

**Prompts**: The model works better with long, detailed prompts (trained on extended prompts). Consider using LLMs to extend short prompts.

**Data-Free Training**: DMD/SID training is data-free (no video data needed), only text prompts. Requires ODE initialization checkpoint.

**Distributed Training**: Uses FSDP (Fully Sharded Data Parallel) with configurable sharding strategies. See `utils/distributed.py` for FSDP wrappers and EMA handling.

**Inference Modes**:
- T2V (text-to-video): Default mode
- I2V (image-to-video): Use `--i2v` flag, requires `TextImagePairDataset`

**Checkpoint Loading**:
- Regular model: `state_dict['generator']`
- EMA model: `state_dict['generator_ema']` (use `--use_ema` flag)

**VAE Decoding**: The VAE maintains cache across blocks for temporal consistency. First block typically skips first 3 decoded frames.

**Torch Compile**: Demo supports `torch.compile` for speedup (compile happens on first block, takes 5-10 min).

## Experiments & Analysis

### Directory Structure
- `experiments/` - Attention extraction and analysis scripts
- `notebooks/` - Jupyter notebooks for visualization
- `figures/` - Generated SVG figures
- `cache/` - Cached attention weights and intermediate data
- `docs/` - 技术文档

### 单层注意力提取

提取单层的完整 frame×frame 注意力矩阵：

```bash
# 提取第 3 层
PYTHONPATH=. python experiments/run_extraction_each.py \
    --layer_index 3 \
    --output_path cache/layer3.pt \
    --checkpoint_path checkpoints/self_forcing_dmd.pt

# 使用原始 Wan 模型
PYTHONPATH=. python experiments/run_extraction_each.py \
    --layer_index 3 \
    --output_path cache/layer3_wan.pt \
    --no_checkpoint
```

**输出数据结构**:
- `full_frame_attention`: [12, 19, 19] - 完整 frame×frame 注意力矩阵
- `last_block_frame_attention`: [12, 19] - 最后 block 对各帧的注意力
- `block_sizes`: [1, 3, 3, 3, 3, 3, 3] - block 结构

**可视化**: `notebooks/extract_all_attention.ipynb`
- 2D 热力图 (Query × Key, 3×4 网格)
- Per-Head Grid 柱状图

### 技术细节

**索引映射**: 每个 Transformer block 有 2 次 attention 调用
- 偶数索引 = self-attention (视频 tokens)
- 奇数索引 = cross-attention (文本 tokens)
- Layer N → self-attn call index 2*N

**Block-based Causality**: 与传统 causal attention 不同
- Block 内所有 Q frames 可以看到该 block 结束为止的所有 K frames
- 例：Block 1 的 Q frames 1-3 都可以看到 K frames 0-3

### 文档

- `docs/attention_extraction.md` - 提取逻辑详解
- `docs/attention_visualization.md` - 绘图逻辑详解
- `docs/testing.md` - 测试模块文档

## Testing

### 运行测试

```bash
# 运行所有单元测试（跳过 GPU 测试，约 35 秒）
python -m pytest tests -v

# 运行 GPU 集成测试（需要 CUDA，约 10 分钟）
python -m pytest tests --run-slow -v

# 只运行特定测试
python -m pytest tests/test_extraction_logic.py -v
```

### 测试结构

```
tests/
├── conftest.py                    # pytest 配置和进度提示 hooks
├── test_extraction_logic.py       # 主测试文件 (6 个测试类)
├── test_attention_equivalence.py  # attention 函数等价性测试
├── test_attention_extraction.py   # 注意力捕获机制测试
└── test_flash.py                  # Flash Attention 测试
```

### 测试标记

| 标记 | 说明 |
|-----|------|
| `@pytest.mark.slow` | GPU 测试，需要 `--run-slow` |
| `@pytest.mark.gpu` | 需要 CUDA |

### 关键测试类

- `TestBlockStructure` - block 结构计算
- `TestIndexMapping` - layer → self-attn 索引映射
- `TestAttentionCaptureMechanism` - ATTENTION_WEIGHT_CAPTURE 机制
- `TestFrameAttentionComputation` - token→frame 聚合
- `TestIntegrationWithGPU` - GPU 集成测试
