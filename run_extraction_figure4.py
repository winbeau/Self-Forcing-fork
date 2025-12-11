#!/usr/bin/env python
"""
Figure 4 注意力权重提取脚本

本脚本通过手动计算注意力权重来获取完整的帧间注意力矩阵，
用于复现论文 Figure 4。

与推理模式不同，这里直接计算 Q @ K^T 得到完整的注意力矩阵，
避免 KV cache 滑动窗口的限制。

内存优化版本：使用分块计算避免 OOM，支持多 GPU。

用法：
    python run_extraction_figure4.py \
        --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --output_path attention_cache_figure4.pt \
        --layer_indices 0 4 \
        --num_frames 21
"""

import argparse
import torch
import os
import sys
import gc

# 延迟导入以避免 CUDA 初始化问题
def main():
    args = parse_args()

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        sys.exit(1)

    # 现在导入需要 CUDA 的模块
    from omegaconf import OmegaConf
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
    from utils.misc import set_seed

    set_seed(args.seed)

    # 选择 GPU
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # 加载配置
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # 初始化模型
    print("Initializing models...")
    torch.set_grad_enabled(False)

    generator = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}),
        is_causal=True
    )
    text_encoder = WanTextEncoder()

    # 加载 checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            generator.load_state_dict(state_dict[key])
        else:
            generator.load_state_dict(state_dict['generator'])
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}")

    # 移到 GPU
    generator = generator.to(device=device, dtype=torch.bfloat16)
    text_encoder = text_encoder.to(device=device)
    generator.eval()

    # 设置模型参数
    num_frame_per_block = config.get('num_frame_per_block', 3)
    generator.model.num_frame_per_block = num_frame_per_block

    print(f"\nNum frames: {args.num_frames}")
    print(f"Layer indices: {args.layer_indices}")
    print(f"Num frame per block: {num_frame_per_block}")

    # 创建输入
    batch_size = 1
    noisy_latent = torch.randn(
        [batch_size, args.num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # 获取文本嵌入
    conditional_dict = text_encoder(text_prompts=[args.prompt])

    # 计算注意力权重
    print("\nComputing attention weights...")
    print(f"Chunk size: {args.chunk_size} (frames per chunk for memory efficiency)")
    attention_data = compute_attention_weights_chunked(
        generator,
        noisy_latent,
        args.layer_indices,
        num_frame_per_block,
        chunk_size=args.chunk_size
    )

    if not attention_data:
        print("Error: No attention data captured!")
        sys.exit(1)

    # 保存
    save_data = {
        'attention_weights': attention_data,
        'prompt': args.prompt,
        'num_frames': args.num_frames,
        'frame_seq_length': attention_data[0]['frame_seq_length'],
        'num_frame_per_block': num_frame_per_block,
        'layer_indices': args.layer_indices,
    }

    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("ATTENTION DATA SUMMARY")
    print("="*60)
    for w in attention_data:
        layer = w['layer_idx']
        shape = tuple(w['attn_weights'].shape)
        num_frames = w['num_frames']
        frame_seq = w['frame_seq_length']
        print(f"Layer {layer}: {shape}")
        print(f"  - {num_frames} frames, {frame_seq} tokens/frame")
        print(f"  - Total tokens: {num_frames * frame_seq}")
        print(f"  - Key frame indices: 0 to {num_frames - 1}")


def compute_attention_weights_chunked(
    generator,
    noisy_latent: torch.Tensor,
    layer_indices: list,
    num_frame_per_block: int,
    chunk_size: int = 3,
):
    """
    分块计算注意力权重，避免 OOM。

    策略：
    - 只计算最后一个 block 的 query 对所有之前 token 的注意力
    - 逐帧分块计算注意力分数，避免一次性分配巨大矩阵
    - 使用 bfloat16 计算以节省内存

    Args:
        generator: WanDiffusionWrapper 模型
        noisy_latent: [B, F, C, H, W] 输入 latent
        layer_indices: 要提取的层索引
        num_frame_per_block: 每个 block 的帧数
        chunk_size: 每次计算多少帧的 key（内存 vs 速度的权衡）
    """
    device = noisy_latent.device
    wan_model = generator.model

    batch_size = noisy_latent.shape[0]
    num_frames = noisy_latent.shape[1]

    # Patch embedding
    x = noisy_latent.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    x = wan_model.patch_embedding(x)  # [B, dim, F', H', W']
    grid_sizes = torch.tensor([x.shape[2:]], dtype=torch.long, device=device)
    x = x.flatten(2).transpose(1, 2)  # [B, seq_len, dim]

    seq_len = x.shape[1]
    frame_seq_length = seq_len // num_frames

    print(f"Input shape after patch embedding: {x.shape}")
    print(f"Frames: {num_frames}, Tokens per frame: {frame_seq_length}")
    print(f"Total sequence length: {seq_len}")

    # 最后一个 block 的起始帧
    last_block_start_frame = ((num_frames - 1) // num_frame_per_block) * num_frame_per_block
    last_block_start_token = last_block_start_frame * frame_seq_length
    last_block_num_tokens = (num_frames - last_block_start_frame) * frame_seq_length

    # 可以 attend 到的范围（最后一个 block 可以看到所有之前的 token）
    last_block_end_frame = min((last_block_start_frame // num_frame_per_block + 1) * num_frame_per_block, num_frames)
    attend_end_token = last_block_end_frame * frame_seq_length

    print(f"\nLast block starts at frame {last_block_start_frame}")
    print(f"Last block tokens: {last_block_start_token} to {seq_len}")
    print(f"Last block can attend to tokens 0 to {attend_end_token}")

    captured_attention = []

    # 遍历 transformer blocks
    for block_idx, block in enumerate(wan_model.blocks):
        if layer_indices is not None and block_idx not in layer_indices:
            continue

        print(f"\nProcessing layer {block_idx}...")

        # 清理之前的内存
        torch.cuda.empty_cache()
        gc.collect()

        # 获取 self-attention 模块
        self_attn = block.self_attn

        # 归一化输入
        x_norm = block.norm1(x)

        # 获取 Q, K, V
        b, s, d = x_norm.shape
        n = self_attn.num_heads
        head_dim = self_attn.head_dim
        scale = head_dim ** -0.5

        print(f"  Heads: {n}, Head dim: {head_dim}")

        # 只计算最后一个 block 的 Q
        x_norm_last_block = x_norm[:, last_block_start_token:, :]
        q_last = self_attn.norm_q(self_attn.q(x_norm_last_block))
        q_last = q_last.view(b, last_block_num_tokens, n, head_dim)
        q_last = q_last.transpose(1, 2)  # [B, N, last_tokens, D]

        # 计算全部的 K（但我们只需要能 attend 到的部分）
        x_norm_attend = x_norm[:, :attend_end_token, :]
        k_attend = self_attn.norm_k(self_attn.k(x_norm_attend))
        k_attend = k_attend.view(b, attend_end_token, n, head_dim)
        k_attend = k_attend.transpose(1, 2)  # [B, N, attend_tokens, D]

        print(f"  Q shape (last block): {q_last.shape}")
        print(f"  K shape (attendable): {k_attend.shape}")

        # 估算内存需求
        # attn_scores: [B, N, last_tokens, attend_tokens] in float32
        attn_mem_gb = (b * n * last_block_num_tokens * attend_end_token * 4) / (1024**3)
        print(f"  Estimated attention matrix size: {attn_mem_gb:.2f} GB")

        if attn_mem_gb > 30:
            print(f"  Using chunked computation (chunk_size={chunk_size} frames)...")
            # 分块计算
            attn_weights = compute_attention_chunked(
                q_last, k_attend, scale,
                frame_seq_length, chunk_size, device
            )
        else:
            # 直接计算
            print(f"  Computing full attention matrix...")
            attn_scores = torch.matmul(
                q_last.float(),
                k_attend.float().transpose(-2, -1)
            ) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            del attn_scores

        print(f"  Final attention shape: {attn_weights.shape}")

        captured_attention.append({
            'layer_idx': block_idx,
            'attn_weights': attn_weights.cpu().to(torch.float16),  # 转为 float16 节省存储
            'num_frames': num_frames,
            'frame_seq_length': frame_seq_length,
            'last_block_start_frame': last_block_start_frame,
            'num_frame_per_block': num_frame_per_block,
            'attend_end_token': attend_end_token,
        })

        # 清理内存
        del q_last, k_attend, attn_weights, x_norm_last_block, x_norm_attend
        torch.cuda.empty_cache()
        gc.collect()

    return captured_attention


def compute_attention_chunked(q, k, scale, frame_seq_length, chunk_size, device):
    """
    分块计算注意力，避免一次性分配大矩阵。

    Args:
        q: [B, N, Q, D] query tensor
        k: [B, N, K, D] key tensor
        scale: attention scale factor
        frame_seq_length: tokens per frame
        chunk_size: number of frames per chunk
        device: computation device

    Returns:
        attn_weights: [B, N, Q, K] attention weights
    """
    b, n, q_len, d = q.shape
    k_len = k.shape[2]

    # 分块计算 attention scores
    chunk_tokens = chunk_size * frame_seq_length
    num_chunks = (k_len + chunk_tokens - 1) // chunk_tokens

    print(f"    Computing attention in {num_chunks} chunks...")

    # 分块计算并找到 max（用于数值稳定的 softmax）
    max_scores = None
    score_chunks = []

    for i in range(num_chunks):
        start = i * chunk_tokens
        end = min((i + 1) * chunk_tokens, k_len)

        k_chunk = k[:, :, start:end, :]
        scores_chunk = torch.matmul(q.float(), k_chunk.float().transpose(-2, -1)) * scale
        score_chunks.append(scores_chunk.cpu())  # 暂存到 CPU

        chunk_max = scores_chunk.max(dim=-1, keepdim=True)[0]
        if max_scores is None:
            max_scores = chunk_max
        else:
            max_scores = torch.maximum(max_scores, chunk_max)

        del scores_chunk, k_chunk
        torch.cuda.empty_cache()

    # 计算 softmax（分块实现）
    # softmax(x) = exp(x - max) / sum(exp(x - max))
    print(f"    Computing softmax...")

    sum_exp = None
    for i, scores_chunk in enumerate(score_chunks):
        scores_chunk = scores_chunk.to(device)
        exp_chunk = torch.exp(scores_chunk - max_scores)

        if sum_exp is None:
            sum_exp = exp_chunk.sum(dim=-1, keepdim=True)
        else:
            sum_exp = sum_exp + exp_chunk.sum(dim=-1, keepdim=True)

        score_chunks[i] = exp_chunk.cpu()  # 保存 exp 结果
        del scores_chunk, exp_chunk
        torch.cuda.empty_cache()

    # 最终归一化
    print(f"    Normalizing attention weights...")
    attn_chunks = []
    for exp_chunk in score_chunks:
        exp_chunk = exp_chunk.to(device)
        attn_chunk = exp_chunk / sum_exp
        attn_chunks.append(attn_chunk.cpu())
        del exp_chunk
        torch.cuda.empty_cache()

    # 拼接结果
    attn_weights = torch.cat(attn_chunks, dim=-1).to(device)

    del score_chunks, attn_chunks, sum_exp, max_scores
    torch.cuda.empty_cache()

    return attn_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Extract attention weights for Figure 4")
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="attention_cache_figure4.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--layer_indices", type=int, nargs='+', default=[0, 4])
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--chunk_size", type=int, default=3,
                        help="Number of frames per chunk for memory-efficient attention computation. "
                             "Lower values use less memory but are slower. Default: 3")
    return parser.parse_args()


if __name__ == "__main__":
    main()
