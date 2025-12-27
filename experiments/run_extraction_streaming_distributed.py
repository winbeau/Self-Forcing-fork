#!/usr/bin/env python
"""
分布式流式帧级注意力提取脚本

支持多种分布式策略：
1. 多层并行：每张卡提取不同层的注意力（推荐，最高效）
2. 多 prompt 并行：每张卡用不同 prompt 生成视频

用法：
    # 4 卡并行提取 4 层（每卡一层）
    torchrun --nproc_per_node=4 experiments/run_extraction_streaming_distributed.py \
        --mode multi_layer \
        --layer_indices 0,8,16,24 \
        --num_frames 60 \
        --output_dir cache/distributed

    # 4 卡并行提取所有 30 层（每卡 7-8 层）
    torchrun --nproc_per_node=4 experiments/run_extraction_streaming_distributed.py \
        --mode multi_layer \
        --all_layers \
        --num_frames 60 \
        --output_dir cache/all_layers

    # 4 卡用不同 prompt 并行
    torchrun --nproc_per_node=4 experiments/run_extraction_streaming_distributed.py \
        --mode multi_prompt \
        --prompts_file prompts/test_prompts.txt \
        --layer_index 15 \
        --num_frames 60 \
        --output_dir cache/multi_prompt
"""

import argparse
import torch
import torch.distributed as dist
import os
import sys
import gc


def setup_distributed():
    """
    初始化分布式环境。

    注意：由于每张卡独立运行推理（不需要真正的分布式通信），
    我们使用 gloo backend 而非 nccl，避免共享内存问题。
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        # 使用 gloo backend，避免 NCCL 共享内存问题
        # 因为我们不需要 GPU 间通信，只需要进程协调
        dist.init_process_group(backend='gloo')

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_multi_layer_extraction(args, rank, world_size, local_rank):
    """
    多层并行提取模式。

    每张卡独立运行推理，但只提取分配给自己的层。
    """
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.attention_distributed import (
        DISTRIBUTED_STREAMING_CAPTURE,
        DistributedFrameAttentionAggregator,
    )
    from utils.misc import set_seed

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 确定每卡负责的层
    if args.all_layers:
        all_layer_indices = list(range(30))
    else:
        all_layer_indices = [int(x) for x in args.layer_indices.split(',')]

    # 分配层到各卡
    layers_per_rank = len(all_layer_indices) // world_size
    remainder = len(all_layer_indices) % world_size

    if rank < remainder:
        start_idx = rank * (layers_per_rank + 1)
        end_idx = start_idx + layers_per_rank + 1
    else:
        start_idx = remainder * (layers_per_rank + 1) + (rank - remainder) * layers_per_rank
        end_idx = start_idx + layers_per_rank

    my_layers = all_layer_indices[start_idx:end_idx]

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DISTRIBUTED MULTI-LAYER EXTRACTION")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Total layers to extract: {len(all_layer_indices)}")
        print(f"Layers: {all_layer_indices}")
        print(f"{'='*60}")

    print(f"[Rank {rank}] Assigned layers: {my_layers}")

    if not my_layers:
        print(f"[Rank {rank}] No layers assigned, exiting")
        return

    set_seed(args.seed + rank)

    # 显存估算
    per_frame_mb = 450
    base_gb = 3.0
    save_overhead_gb = 6.0
    runtime_gb = base_gb + args.num_frames * per_frame_mb / 1024
    peak_gb = runtime_gb + save_overhead_gb
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3

    if rank == 0:
        print(f"\nMemory estimation for {args.num_frames} frames:")
        print(f"  Runtime: ~{runtime_gb:.1f} GB")
        print(f"  Peak: ~{peak_gb:.1f} GB")
        print(f"  GPU available: {total_mem:.1f} GB")

    if peak_gb > total_mem * 0.95 and not args.force:
        print(f"[Rank {rank}] ⚠️ Peak memory ({peak_gb:.1f} GB) may exceed GPU ({total_mem:.1f} GB)!")
        sys.exit(1)

    # 加载配置和模型
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if rank == 0:
        print("\nInitializing inference pipeline...")

    torch.set_grad_enabled(False)
    pipeline = CausalInferencePipeline(args=config, device=device)

    # 加载 checkpoint
    if args.no_checkpoint:
        if rank == 0:
            print("Using original Wan2.1 base model (no checkpoint)")
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        if rank == 0:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])
    else:
        if rank == 0:
            print("Warning: checkpoint not found, using original Wan2.1 base model")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    num_frame_per_block = config.get('num_frame_per_block', 3)
    num_frames = args.num_frames
    frame_seq_length = 1560

    # 配置 KV cache
    required_kv_cache_size = num_frames * frame_seq_length

    for block in pipeline.generator.model.blocks:
        if hasattr(block, 'self_attn'):
            block.self_attn.max_attention_size = required_kv_cache_size
            block.self_attn.local_attn_size = num_frames

    # 初始化 KV cache
    num_transformer_blocks = 30
    kv_cache = []
    for _ in range(num_transformer_blocks):
        kv_cache.append({
            "k": torch.zeros([1, required_kv_cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, required_kv_cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        })
    pipeline.kv_cache1 = kv_cache

    crossattn_cache = []
    for _ in range(num_transformer_blocks):
        crossattn_cache.append({
            "k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "is_init": False
        })
    pipeline.crossattn_cache = crossattn_cache

    # 计算 block 结构
    independent_first_frame = config.get('independent_first_frame', True)
    if independent_first_frame:
        block_sizes = [1] + [num_frame_per_block] * ((num_frames - 1) // num_frame_per_block)
    else:
        block_sizes = [num_frame_per_block] * (num_frames // num_frame_per_block)

    actual_frames = sum(block_sizes)
    if actual_frames != num_frames:
        if rank == 0:
            print(f"Adjusting num_frames from {num_frames} to {actual_frames}")
        num_frames = actual_frames

    # 创建噪声
    noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    num_layers = len(pipeline.generator.model.blocks)

    # 对每个分配的层进行提取
    os.makedirs(args.output_dir, exist_ok=True)

    for layer_index in my_layers:
        print(f"\n[Rank {rank}] Extracting layer {layer_index}...")

        # 重置 KV cache
        for kv in pipeline.kv_cache1:
            kv["k"].zero_()
            kv["v"].zero_()
            kv["global_end_index"].zero_()
            kv["local_end_index"].zero_()

        for ca in pipeline.crossattn_cache:
            ca["k"].zero_()
            ca["v"].zero_()
            ca["is_init"] = False

        # 创建聚合器
        aggregator = DistributedFrameAttentionAggregator(
            num_frames=num_frames,
            num_heads=12,
            block_sizes=block_sizes,
            rank=rank,
            world_size=world_size,
        )

        # Layer N → self-attn call index 2*N
        self_attn_idx = 2 * layer_index

        # 启用分布式捕获
        DISTRIBUTED_STREAMING_CAPTURE.enable(
            layer_indices=[self_attn_idx],
            num_layers=num_layers * 2,
            frame_seq_length=frame_seq_length,
            num_heads=12,
            chunk_frames=args.chunk_frames,
            total_frames=num_frames,
            on_frame_attention=aggregator.on_frame_attention,
        )

        try:
            output = pipeline.inference(
                noise=noise.clone(),
                text_prompts=[args.prompt],
                return_latents=True,
            )
        finally:
            DISTRIBUTED_STREAMING_CAPTURE.disable()

        # 获取结果
        full_frame_attn = aggregator.full_frame_attn

        # 计算最后一个 block 的 per-head 帧注意力
        last_block_q_start = sum(block_sizes[:-1])
        last_block_frame_attn = full_frame_attn[:, last_block_q_start:, :].mean(dim=1)

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
            'extraction_method': 'distributed_streaming',
            'rank': rank,
            'world_size': world_size,
        }

        output_path = os.path.join(args.output_dir, f"layer{layer_index}.pt")
        torch.save(save_data, output_path)
        print(f"[Rank {rank}] Saved layer {layer_index} to {output_path}")

        # 清理
        del aggregator, full_frame_attn
        gc.collect()
        torch.cuda.empty_cache()

    # 注意：由于每张卡独立运行，不需要 barrier 同步
    # 每张卡完成后直接退出即可

    if rank == 0:
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Output directory: {args.output_dir}")
        print(f"Total layers extracted: {len(all_layer_indices)}")


def run_multi_prompt_extraction(args, rank, world_size, local_rank):
    """
    多 prompt 并行提取模式。

    每张卡用不同的 prompt 生成视频并提取注意力。
    """
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.attention_distributed import (
        DISTRIBUTED_STREAMING_CAPTURE,
        DistributedFrameAttentionAggregator,
    )
    from utils.misc import set_seed

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 读取 prompts
    with open(args.prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    # 分配 prompts 到各卡
    prompts_per_rank = len(all_prompts) // world_size
    remainder = len(all_prompts) % world_size

    if rank < remainder:
        start_idx = rank * (prompts_per_rank + 1)
        end_idx = start_idx + prompts_per_rank + 1
    else:
        start_idx = remainder * (prompts_per_rank + 1) + (rank - remainder) * prompts_per_rank
        end_idx = start_idx + prompts_per_rank

    my_prompts = all_prompts[start_idx:end_idx]
    my_prompt_indices = list(range(start_idx, end_idx))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DISTRIBUTED MULTI-PROMPT EXTRACTION")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Total prompts: {len(all_prompts)}")
        print(f"Layer to extract: {args.layer_index}")
        print(f"{'='*60}")

    print(f"[Rank {rank}] Assigned prompts: {len(my_prompts)} (indices {start_idx}-{end_idx-1})")

    if not my_prompts:
        print(f"[Rank {rank}] No prompts assigned, exiting")
        return

    set_seed(args.seed + rank)

    # 加载配置和模型（与 multi_layer 类似）
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    torch.set_grad_enabled(False)
    pipeline = CausalInferencePipeline(args=config, device=device)

    if args.no_checkpoint:
        pass
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    num_frame_per_block = config.get('num_frame_per_block', 3)
    num_frames = args.num_frames
    frame_seq_length = 1560

    # 配置 KV cache
    required_kv_cache_size = num_frames * frame_seq_length

    for block in pipeline.generator.model.blocks:
        if hasattr(block, 'self_attn'):
            block.self_attn.max_attention_size = required_kv_cache_size
            block.self_attn.local_attn_size = num_frames

    num_transformer_blocks = 30
    kv_cache = []
    for _ in range(num_transformer_blocks):
        kv_cache.append({
            "k": torch.zeros([1, required_kv_cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, required_kv_cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        })
    pipeline.kv_cache1 = kv_cache

    crossattn_cache = []
    for _ in range(num_transformer_blocks):
        crossattn_cache.append({
            "k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
            "is_init": False
        })
    pipeline.crossattn_cache = crossattn_cache

    # 计算 block 结构
    independent_first_frame = config.get('independent_first_frame', True)
    if independent_first_frame:
        block_sizes = [1] + [num_frame_per_block] * ((num_frames - 1) // num_frame_per_block)
    else:
        block_sizes = [num_frame_per_block] * (num_frames // num_frame_per_block)

    actual_frames = sum(block_sizes)
    if actual_frames != num_frames:
        num_frames = actual_frames

    num_layers = len(pipeline.generator.model.blocks)
    layer_index = args.layer_index
    self_attn_idx = 2 * layer_index

    os.makedirs(args.output_dir, exist_ok=True)

    # 对每个 prompt 进行提取
    for prompt_idx, (global_idx, prompt) in enumerate(zip(my_prompt_indices, my_prompts)):
        print(f"\n[Rank {rank}] Processing prompt {global_idx}: {prompt[:50]}...")

        # 重置 KV cache
        for kv in pipeline.kv_cache1:
            kv["k"].zero_()
            kv["v"].zero_()
            kv["global_end_index"].zero_()
            kv["local_end_index"].zero_()

        for ca in pipeline.crossattn_cache:
            ca["k"].zero_()
            ca["v"].zero_()
            ca["is_init"] = False

        # 创建噪声（每个 prompt 使用不同的随机种子）
        torch.manual_seed(args.seed + global_idx)
        noise = torch.randn(
            [1, num_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
        )

        aggregator = DistributedFrameAttentionAggregator(
            num_frames=num_frames,
            num_heads=12,
            block_sizes=block_sizes,
            rank=rank,
            world_size=world_size,
        )

        DISTRIBUTED_STREAMING_CAPTURE.enable(
            layer_indices=[self_attn_idx],
            num_layers=num_layers * 2,
            frame_seq_length=frame_seq_length,
            num_heads=12,
            chunk_frames=args.chunk_frames,
            total_frames=num_frames,
            on_frame_attention=aggregator.on_frame_attention,
        )

        try:
            output = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                return_latents=True,
            )
        finally:
            DISTRIBUTED_STREAMING_CAPTURE.disable()

        full_frame_attn = aggregator.full_frame_attn
        last_block_q_start = sum(block_sizes[:-1])
        last_block_frame_attn = full_frame_attn[:, last_block_q_start:, :].mean(dim=1)

        save_data = {
            'layer_index': layer_index,
            'full_frame_attention': full_frame_attn.to(torch.float16),
            'last_block_frame_attention': last_block_frame_attn.to(torch.float16),
            'is_logits': True,
            'prompt': prompt,
            'prompt_index': global_idx,
            'num_frames': num_frames,
            'frame_seq_length': frame_seq_length,
            'num_frame_per_block': num_frame_per_block,
            'num_heads': 12,
            'block_sizes': block_sizes,
            'extraction_method': 'distributed_multi_prompt',
        }

        # 使用 prompt index 命名
        output_path = os.path.join(args.output_dir, f"prompt{global_idx:04d}_layer{layer_index}.pt")
        torch.save(save_data, output_path)
        print(f"[Rank {rank}] Saved to {output_path}")

        del aggregator, full_frame_attn
        gc.collect()
        torch.cuda.empty_cache()

    # 注意：由于每张卡独立运行，不需要 barrier 同步

    if rank == 0:
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed streaming attention extraction")

    # 模式选择
    parser.add_argument("--mode", type=str, choices=['multi_layer', 'multi_prompt'],
                        default='multi_layer', help="Extraction mode")

    # 通用参数
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_dir", type=str, default="cache/distributed")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--chunk_frames", type=int, default=3)
    parser.add_argument("--no_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force run even if memory warning")

    # multi_layer 模式参数
    parser.add_argument("--layer_indices", type=str, default="0,8,16,24",
                        help="Comma-separated layer indices to extract")
    parser.add_argument("--all_layers", action="store_true",
                        help="Extract all 30 layers")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")

    # multi_prompt 模式参数
    parser.add_argument("--prompts_file", type=str, default="prompts/test_prompts.txt",
                        help="File containing prompts (one per line)")
    parser.add_argument("--layer_index", type=int, default=15,
                        help="Layer index to extract (for multi_prompt mode)")

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        sys.exit(1)

    rank, world_size, local_rank = setup_distributed()

    try:
        if args.mode == 'multi_layer':
            run_multi_layer_extraction(args, rank, world_size, local_rank)
        elif args.mode == 'multi_prompt':
            run_multi_prompt_extraction(args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
