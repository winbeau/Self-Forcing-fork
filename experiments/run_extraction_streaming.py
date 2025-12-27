#!/usr/bin/env python
"""
流式帧级注意力提取脚本

使用 STREAMING_FRAME_ATTENTION_CAPTURE 进行内存高效的注意力提取。
支持高帧数（100+），不会 OOM。

用法：
    PYTHONPATH=. python experiments/run_extraction_streaming.py \
        --layer_index 15 \
        --num_frames 120 \
        --output_path cache/layer15_120frames.pt

    # 测试极限
    PYTHONPATH=. python experiments/run_extraction_streaming.py \
        --layer_index 15 \
        --num_frames 200 \
        --output_path cache/layer15_200frames.pt
"""

import argparse
import torch
import os
import sys
import gc


class FrameAttentionAggregator:
    """
    帧级注意力聚合器：收集流式回调的数据并构建完整的 frame×frame 矩阵。
    """

    def __init__(self, num_frames, num_heads, block_sizes):
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.block_sizes = block_sizes

        # 完整的 frame-level attention: [num_heads, num_frames, num_frames]
        self.full_frame_attn = torch.zeros(
            num_heads, num_frames, num_frames,
            dtype=torch.float32
        )

        # 追踪处理进度
        self.last_k_frames = 0
        self.current_q_start = 0
        self.blocks_processed = 0

    def on_frame_attention(self, layer_idx, frame_attn, q_frames, k_frames):
        """
        回调函数：接收流式的 frame-level attention 数据。

        Args:
            layer_idx: 层索引
            frame_attn: [num_heads, q_frames, k_frames] 的注意力数据
            q_frames: Query 帧数
            k_frames: Key 帧数
        """
        # 只处理新的 block（K frames 增加时）
        if k_frames <= self.last_k_frames:
            return

        if self.blocks_processed >= len(self.block_sizes):
            return

        block_size = self.block_sizes[self.blocks_processed]

        print(f"  Block {self.blocks_processed}: Q frames {self.current_q_start}-{self.current_q_start + q_frames - 1}, "
              f"K frames 0-{k_frames - 1}")

        # 将数据填入完整矩阵
        q_end = self.current_q_start + q_frames
        self.full_frame_attn[:, self.current_q_start:q_end, :k_frames] = frame_attn

        # 更新状态
        self.last_k_frames = k_frames
        self.current_q_start += block_size
        self.blocks_processed += 1


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        sys.exit(1)

    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.attention import STREAMING_FRAME_ATTENTION_CAPTURE
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

    # ========== 显存估算（基于实测校准） ==========
    #
    # 实测校准 (60帧):
    #   - 运行时稳定: 30 GB
    #   - 保存峰值: 36 GB
    #
    # 线性模型: runtime = base + frames × per_frame
    #   - base ≈ 3 GB (模型权重)
    #   - per_frame = (30 - 3) / 60 = 0.45 GB = 450 MB
    #
    # 注: 实测值高于理论 KV cache (274 MB/frame)，
    #     额外开销来自中间激活、推理临时 tensor 等
    #
    chunk_frames = args.chunk_frames

    # 基于实测的每帧显存占用
    per_frame_memory_mb = 450  # 实测校准值

    # Chunk attention 临时显存
    chunk_attn_mb = 12 * 1560 * (chunk_frames * 1560) * 4 / 1024**2

    # 基础占用 (模型权重 + cross-attn cache + 其他固定开销)
    base_memory_gb = 3.0

    # 运行时稳定显存
    runtime_estimated_gb = base_memory_gb + num_frames * per_frame_memory_mb / 1024 + chunk_attn_mb / 1024

    # 保存时峰值显存 (序列化、类型转换、临时缓冲)
    save_overhead_gb = 6.0  # 实测保存时额外占用约 6GB
    peak_estimated_gb = runtime_estimated_gb + save_overhead_gb

    print(f"\n{'='*60}")
    print(f"STREAMING EXTRACTION (Memory Efficient)")
    print(f"{'='*60}")
    print(f"Memory estimation for {num_frames} frames:")
    print(f"  Per-frame cost: {per_frame_memory_mb} MB (calibrated)")
    print(f"  Frames memory: {num_frames * per_frame_memory_mb / 1024:.1f} GB")
    print(f"  Chunk attention ({chunk_frames} frames): {chunk_attn_mb / 1024:.2f} GB")
    print(f"  Base (model + misc): {base_memory_gb:.1f} GB")
    print(f"  Runtime estimated: ~{runtime_estimated_gb:.1f} GB")
    print(f"  Peak (during save): ~{peak_estimated_gb:.1f} GB")
    print(f"  GPU available: {total_mem:.1f} GB")
    print(f"{'='*60}")

    if peak_estimated_gb > total_mem * 0.95:
        print(f"\n⚠️  Peak memory ({peak_estimated_gb:.1f} GB) may exceed GPU ({total_mem:.1f} GB)!")
        print(f"   Try reducing num_frames or use --force to proceed")
        if not args.force:
            sys.exit(1)

    # ========== 配置 KV cache 和 max_attention_size ==========
    required_kv_cache_size = num_frames * frame_seq_length
    print(f"\nConfiguring for {num_frames} frames...")

    # 更新所有 attention 层的 max_attention_size
    for block in pipeline.generator.model.blocks:
        if hasattr(block, 'self_attn'):
            block.self_attn.max_attention_size = required_kv_cache_size
            block.self_attn.local_attn_size = num_frames

    # 初始化 KV cache
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

    # 初始化 cross-attention cache
    crossattn_cache = []
    for _ in range(num_transformer_blocks):
        crossattn_cache.append({
            "k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "is_init": False
        })
    pipeline.crossattn_cache = crossattn_cache

    print(f"KV cache initialized: {required_kv_cache_size} tokens")

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
        print(f"Adjusting num_frames from {num_frames} to {actual_frames}")
        num_frames = actual_frames

    print(f"\nLayer to capture: {layer_index}")
    print(f"Num frames: {num_frames}")
    print(f"Block structure: {len(block_sizes)} blocks")
    print(f"Chunk size: {chunk_frames} frames")
    print(f"Prompt: {args.prompt[:50]}...")

    # 创建噪声
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

    # Layer N → self-attn call index 2*N
    self_attn_idx = 2 * layer_index
    print(f"Layer {layer_index} -> self-attn call index: {self_attn_idx}")

    # 创建聚合器
    aggregator = FrameAttentionAggregator(
        num_frames=num_frames,
        num_heads=12,
        block_sizes=block_sizes,
    )

    # 启用流式捕获
    STREAMING_FRAME_ATTENTION_CAPTURE.enable(
        layer_indices=[self_attn_idx],
        num_layers=num_layers * 2,
        frame_seq_length=frame_seq_length,
        num_heads=12,
        chunk_frames=chunk_frames,
        on_frame_attention=aggregator.on_frame_attention,
    )

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
            output_video = output[0]  # 已解码的视频 [B, T, C, H, W], range [0, 1]
            output_latent = output[1]
        else:
            output_video = None
            output_latent = output
        print(f"Output latent shape: {output_latent.shape}")
        if output_video is not None:
            print(f"Output video shape: {output_video.shape}")

    finally:
        STREAMING_FRAME_ATTENTION_CAPTURE.disable()

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
        'extraction_method': 'streaming',
        'chunk_frames': chunk_frames,
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
    print(f"Extraction method: streaming (chunk_frames={chunk_frames})")

    # 分析对角线和 sink
    diag_mean = torch.diagonal(full_frame_attn, dim1=1, dim2=2).mean().item()
    first_col = full_frame_attn[:, :, 0]
    first_col_mean = first_col[first_col != 0].mean().item() if (first_col != 0).any() else 0
    print(f"Diagonal mean: {diag_mean:.4f}")
    print(f"First frame (sink) mean: {first_col_mean:.4f}")

    # ========== 保存视频（可选） ==========
    if args.save_video:
        print("\n" + "=" * 60)
        print("SAVING VIDEO")
        print("=" * 60)

        # 确定视频路径
        if args.video_path:
            video_path = args.video_path
        else:
            video_path = args.output_path.replace('.pt', '.mp4')

        if output_video is None:
            print("Error: No video output from pipeline")
        else:
            # output_video: [B, T, C, H, W], range [0, 1]
            # 转换为 [T, H, W, C] 格式，range [0, 255]
            from einops import rearrange
            video = rearrange(output_video, 'b t c h w -> b t h w c').cpu()
            video = (video[0] * 255.0).to(torch.uint8)  # [T, H, W, C]

            print(f"Video shape: {video.shape} [T, H, W, C]")

            # 保存视频
            os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else '.', exist_ok=True)

            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = video.shape[1], video.shape[2]
            out = cv2.VideoWriter(video_path, fourcc, 16, (w, h))
            for frame in video.numpy():
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

            print(f"Video saved to: {video_path}")

        # 清理 VAE cache
        pipeline.vae.model.clear_cache()

        # 清理
        del output_video
        gc.collect()
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming frame-level attention extraction")
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="cache/attention_streaming.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=120,
                        help="Number of frames (can go 100+ with streaming)")
    parser.add_argument("--layer_index", type=int, default=15, help="Layer index to capture (0-29)")
    parser.add_argument("--chunk_frames", type=int, default=3,
                        help="K frames per chunk (smaller = less memory, slower)")
    parser.add_argument("--no_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Force run even if memory warning")
    # 视频保存选项
    parser.add_argument("--save_video", action="store_true", help="Save generated video")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Video output path (default: same as output_path but .mp4)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
