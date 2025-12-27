#!/usr/bin/env python
"""
高帧数注意力提取脚本（优化版）

通过流式聚合减少显存占用，支持更高帧数。
核心优化：捕获后立即聚合到 frame-level 并释放原始 tensor。

用法：
    PYTHONPATH=. python experiments/run_extraction_highframe.py \
        --layer_index 15 \
        --num_frames 120 \
        --output_path cache/layer15_120frames.pt

    # 测试极限
    PYTHONPATH=. python experiments/run_extraction_highframe.py \
        --layer_index 15 \
        --num_frames 200 \
        --output_path cache/layer15_200frames.pt \
        --no_checkpoint
"""

import argparse
import torch
import os
import sys
import gc


class StreamingList(list):
    """自定义 list，支持在 append 时触发回调。"""

    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback

    def append(self, item):
        if self.callback:
            self.callback(item)
        else:
            super().append(item)


class StreamingFrameAggregator:
    """
    流式帧级聚合器：在每个 attention 被捕获后立即聚合到 frame-level。

    这样可以避免存储完整的 token-level attention 矩阵，大幅减少显存。
    """

    def __init__(self, num_frames, num_heads, frame_seq_length, block_sizes, device='cpu'):
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.frame_seq_length = frame_seq_length
        self.block_sizes = block_sizes
        self.device = device

        # 存储 frame-level attention [num_heads, num_frames, num_frames]
        self.full_frame_attn = torch.zeros(
            num_heads, num_frames, num_frames,
            dtype=torch.float32, device=device
        )

        # 追踪处理进度
        self.last_k_frames = 0  # 上一次处理的 K frames 数量
        self.current_q_start = 0
        self.blocks_processed = 0

    def process_attention(self, attn_data):
        """
        处理一个 block 的 attention，立即聚合到 frame-level。

        Args:
            attn_data: dict with 'attn_weights' [1, num_heads, Lq, Lk]
        """
        # 提取 attention logits
        attn_weights = attn_data.get('attn_weights')
        if attn_weights is None:
            return

        k_tokens = attn_weights.shape[-1]
        k_frames_total = k_tokens // self.frame_seq_length

        # 只有当 K frames 增加时才处理（表示新的 block）
        if k_frames_total <= self.last_k_frames:
            return

        # 检查是否已处理完所有 blocks
        if self.blocks_processed >= len(self.block_sizes):
            return

        block_size = self.block_sizes[self.blocks_processed]
        attn_logits = attn_weights[0].float()  # [num_heads, Lq, Lk]

        q_tokens = attn_logits.shape[1]
        q_frames_in_block = q_tokens // self.frame_seq_length

        print(f"  Block {self.blocks_processed}: Q frames {self.current_q_start}-{self.current_q_start + q_frames_in_block - 1}, "
              f"K frames 0-{k_frames_total - 1}, shape {attn_logits.shape}")

        # 聚合到 frame-level
        for h in range(self.num_heads):
            head_attn = attn_logits[h]  # [Lq, Lk]

            for qf_local in range(q_frames_in_block):
                qf_global = self.current_q_start + qf_local
                q_start_tok = qf_local * self.frame_seq_length
                q_end_tok = (qf_local + 1) * self.frame_seq_length

                for kf in range(k_frames_total):
                    k_start_tok = kf * self.frame_seq_length
                    k_end_tok = (kf + 1) * self.frame_seq_length

                    # 平均所有 token pair 的 attention
                    frame_attn_val = head_attn[q_start_tok:q_end_tok, k_start_tok:k_end_tok].mean()
                    self.full_frame_attn[h, qf_global, kf] = frame_attn_val.item()

        # 更新状态
        self.last_k_frames = k_frames_total
        self.current_q_start += block_size
        self.blocks_processed += 1

        # 清理 GPU 内存
        del attn_logits
        torch.cuda.empty_cache()
        gc.collect()


def setup_streaming_capture(aggregator):
    """
    设置流式捕获：用 StreamingList 替换 captured_weights。
    """
    from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE

    # 保存原始列表
    original_list = ATTENTION_WEIGHT_CAPTURE.captured_weights

    def streaming_callback(data):
        # 立即处理并聚合
        aggregator.process_attention(data)
        # 释放原始 attention weights
        if 'attn_weights' in data:
            del data['attn_weights']
        torch.cuda.empty_cache()
        gc.collect()

    # 替换为 StreamingList
    ATTENTION_WEIGHT_CAPTURE.captured_weights = StreamingList(callback=streaming_callback)

    return original_list


def restore_attention_capture(original_list):
    """恢复原始的 captured_weights 列表。"""
    from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE
    ATTENTION_WEIGHT_CAPTURE.captured_weights = original_list


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        sys.exit(1)

    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE
    from utils.misc import set_seed

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # 打印 GPU 信息
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"Total GPU memory: {total_mem:.1f} GB")

    # 加载配置
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    print("\nInitializing inference pipeline...")
    torch.set_grad_enabled(False)

    pipeline = CausalInferencePipeline(args=config, device=device)

    # 加载 checkpoint
    if args.no_checkpoint:
        print("Using original Wan2.1 base model (no checkpoint)")
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])
    else:
        print("Warning: checkpoint not found, using original Wan2.1 base model")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    num_frame_per_block = config.get('num_frame_per_block', 3)
    num_frames = args.num_frames
    frame_seq_length = 1560
    layer_index = args.layer_index

    # ========== 显存估算 ==========
    # KV cache: frames × 1560 × 30 blocks × 12 heads × 128 dim × 2 (k+v) × 2 bytes
    kv_cache_per_frame_mb = 1560 * 30 * 12 * 128 * 2 * 2 / 1024**2
    # Attention (attention_with_weights 需要完整矩阵，随 K 增长):
    # 最后一个 block: 12 heads × 4680 Q × (frames × 1560 K) × 4 bytes × 2 (scores + weights)
    attn_peak_mb = 12 * 4680 * (num_frames * 1560) * 4 * 2 / 1024**2

    total_estimated = 3 + num_frames * kv_cache_per_frame_mb / 1024 + attn_peak_mb / 1024
    print(f"\n{'='*60}")
    print(f"Memory estimation for {num_frames} frames:")
    print(f"  KV cache: {num_frames * kv_cache_per_frame_mb / 1024:.1f} GB")
    print(f"  Peak attention (last block): {attn_peak_mb / 1024:.1f} GB")
    print(f"  Model: ~3 GB")
    print(f"  Total estimated: ~{total_estimated:.1f} GB")
    print(f"  GPU available: {total_mem:.1f} GB")
    print(f"{'='*60}")

    if total_estimated > total_mem * 0.95:
        # 粗略估算最大帧数
        max_frames = int(((total_mem * 0.8 - 3) * 1024 - 500) / (kv_cache_per_frame_mb + 700))
        print(f"\n⚠️  Estimated memory ({total_estimated:.1f} GB) exceeds GPU ({total_mem:.1f} GB)!")
        print(f"   Recommended max frames: {max_frames}")
        if not args.force:
            sys.exit(1)

    # ========== 关键修复：动态调整 KV cache 和 max_attention_size ==========
    required_kv_cache_size = num_frames * frame_seq_length
    print(f"\nAdjusting KV cache size: {required_kv_cache_size} tokens ({num_frames} frames)")

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()

    # 关键：更新所有 attention 层的 max_attention_size
    print(f"Updating max_attention_size on all {len(pipeline.generator.model.blocks)} transformer blocks...")
    for block in pipeline.generator.model.blocks:
        if hasattr(block, 'self_attn'):
            block.self_attn.max_attention_size = required_kv_cache_size
            block.self_attn.local_attn_size = num_frames
    print(f"Set max_attention_size = {required_kv_cache_size}")

    # 直接手动初始化正确大小的 KV cache（绕过 pipeline 的默认逻辑）
    print(f"Manually initializing KV cache with size {required_kv_cache_size}...")
    kv_cache = []
    num_transformer_blocks = 30
    for _ in range(num_transformer_blocks):
        kv_cache.append({
            "k": torch.zeros([1, required_kv_cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, required_kv_cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        })
    pipeline.kv_cache1 = kv_cache

    # 同样初始化 cross-attention cache
    crossattn_cache = []
    for _ in range(num_transformer_blocks):
        crossattn_cache.append({
            "k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "is_init": False
        })
    pipeline.crossattn_cache = crossattn_cache

    print(f"KV cache initialized: {len(kv_cache)} blocks, each with K/V shape [1, {required_kv_cache_size}, 12, 128]")

    # 计算 block 结构
    independent_first_frame = config.get('independent_first_frame', True)
    if independent_first_frame:
        num_blocks = (num_frames - 1) // num_frame_per_block + 1
        block_sizes = [1] + [num_frame_per_block] * ((num_frames - 1) // num_frame_per_block)
    else:
        num_blocks = num_frames // num_frame_per_block
        block_sizes = [num_frame_per_block] * num_blocks

    # 确保帧数与 block 结构匹配
    actual_frames = sum(block_sizes)
    if actual_frames != num_frames:
        print(f"Adjusting num_frames from {num_frames} to {actual_frames} to match block structure")
        num_frames = actual_frames

    print(f"\n{'='*60}")
    print(f"HIGH FRAME EXTRACTION (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Layer to capture: {layer_index}")
    print(f"Num frames: {num_frames}")
    print(f"Block structure: {len(block_sizes)} blocks, sizes: {block_sizes[:5]}...{block_sizes[-2:]}")
    print(f"Prompt: {args.prompt[:50]}...")

    # 创建输入噪声
    noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # 获取模型层数
    num_layers = len(pipeline.generator.model.blocks)
    print(f"Model layers: {num_layers}")

    if layer_index >= num_layers:
        print(f"Error: layer_index {layer_index} >= num_layers {num_layers}")
        sys.exit(1)

    # layer N → 调用索引 2*N (self-attention)
    self_attn_idx = 2 * layer_index
    print(f"Layer {layer_index} -> self-attn call index: {self_attn_idx}")

    # 创建流式聚合器
    aggregator = StreamingFrameAggregator(
        num_frames=num_frames,
        num_heads=12,  # Wan 模型有 12 个 heads
        frame_seq_length=frame_seq_length,
        block_sizes=block_sizes,
        device='cpu'  # 存储在 CPU 上节省 GPU 内存
    )

    # 启用 attention 捕获
    ATTENTION_WEIGHT_CAPTURE.enable(
        layer_indices=[self_attn_idx],
        capture_logits=True,
        num_layers=num_layers * 2
    )

    # 应用流式捕获
    original_list = setup_streaming_capture(aggregator)

    try:
        print("\nRunning inference...")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        output = pipeline.inference(
            noise=noise,
            text_prompts=[args.prompt],
            return_latents=True,
        )

        print(f"Final GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        if isinstance(output, tuple):
            output_latent = output[1]
        else:
            output_latent = output
        print(f"Output latent shape: {output_latent.shape}")

    finally:
        # 恢复原始列表
        restore_attention_capture(original_list)
        ATTENTION_WEIGHT_CAPTURE.disable()

    # 获取结果
    full_frame_attn = aggregator.full_frame_attn

    print(f"\nProcessed {aggregator.blocks_processed} blocks")
    print(f"Full attention matrix shape: {full_frame_attn.shape}")
    print(f"Range: [{full_frame_attn.min():.4f}, {full_frame_attn.max():.4f}]")

    # 计算最后一个 block 的 per-head 帧注意力
    last_block_q_start = sum(block_sizes[:-1])
    last_block_q_end = num_frames
    last_block_q_frames = list(range(last_block_q_start, last_block_q_end))
    last_block_frame_attn = full_frame_attn[:, last_block_q_start:last_block_q_end, :].mean(dim=1)

    # 保存数据
    save_data = {
        'layer_index': layer_index,
        'full_frame_attention': full_frame_attn.to(torch.float16),
        'last_block_frame_attention': last_block_frame_attn.to(torch.float16),
        'is_logits': True,
        'prompt': args.prompt,
        'num_frames': num_frames,
        'frame_seq_length': frame_seq_length,
        'num_frame_per_block': num_frame_per_block,
        'num_heads': 12,
        'block_sizes': block_sizes,
        'query_frames': list(range(num_frames)),
        'key_frames': list(range(num_frames)),
        'last_block_query_frames': last_block_q_frames,
    }

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Layer: {layer_index}")
    print(f"Frames: {num_frames}")
    print(f"Full attention: {tuple(full_frame_attn.shape)} [num_heads, Q_frames, K_frames]")
    print(f"Last block attention: {tuple(last_block_frame_attn.shape)} [num_heads, K_frames]")

    # 分析对角线和 sink
    diag_mean = torch.diagonal(full_frame_attn, dim1=1, dim2=2).mean().item()
    first_col = full_frame_attn[:, :, 0]
    first_col_mean = first_col[first_col != 0].mean().item() if (first_col != 0).any() else 0
    print(f"Diagonal mean: {diag_mean:.4f}")
    print(f"First frame (sink) mean: {first_col_mean:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="High-frame attention extraction (optimized)")
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="cache/attention_highframe.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=27,
                        help="Number of frames (L40 max ~24-27 with attention_with_weights)")
    parser.add_argument("--layer_index", type=int, default=15, help="Layer index to capture (0-29)")
    parser.add_argument("--no_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Force run even if memory warning")
    return parser.parse_args()


if __name__ == "__main__":
    main()
