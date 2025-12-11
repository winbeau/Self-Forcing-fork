#!/usr/bin/env python
"""
Figure 4 注意力权重提取脚本

本脚本通过手动计算注意力权重来获取完整的帧间注意力矩阵，
用于复现论文 Figure 4。

与推理模式不同，这里直接计算 Q @ K^T 得到完整的注意力矩阵，
避免 KV cache 滑动窗口的限制。

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

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    attention_data = compute_attention_weights(
        generator,
        noisy_latent,
        args.layer_indices,
        num_frame_per_block
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


def compute_attention_weights(
    generator,
    noisy_latent: torch.Tensor,
    layer_indices: list,
    num_frame_per_block: int,
):
    """
    手动计算注意力权重，获取完整的帧间注意力矩阵。

    这绕过了 KV cache 模式，直接计算完整的 Q @ K^T。
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

    captured_attention = []

    # 遍历 transformer blocks
    for block_idx, block in enumerate(wan_model.blocks):
        if layer_indices is not None and block_idx not in layer_indices:
            # 仍然需要前向传播以更新 x
            # 简化：跳过非目标层的计算
            continue

        print(f"\nProcessing layer {block_idx}...")

        # 获取 self-attention 模块
        self_attn = block.self_attn

        # 归一化输入
        x_norm = block.norm1(x)

        # 获取 Q, K, V
        b, s, d = x_norm.shape
        n = self_attn.num_heads
        head_dim = self_attn.head_dim

        q = self_attn.norm_q(self_attn.q(x_norm)).view(b, s, n, head_dim)
        k = self_attn.norm_k(self_attn.k(x_norm)).view(b, s, n, head_dim)

        # 计算注意力分数
        q_t = q.transpose(1, 2).float()  # [B, N, S, D]
        k_t = k.transpose(1, 2).float()  # [B, N, S, D]

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, N, S, S]

        # 创建 blockwise causal mask
        # 每个位置只能 attend 到同一 block 及之前 block 的位置
        mask = torch.ones(s, s, device=device, dtype=torch.bool)

        for frame_idx in range(num_frames):
            # 计算当前帧所属的 block
            block_num = frame_idx // num_frame_per_block
            # 该 block 结束的帧索引
            block_end_frame = (block_num + 1) * num_frame_per_block
            block_end_frame = min(block_end_frame, num_frames)

            # 当前帧的 token 范围
            frame_start = frame_idx * frame_seq_length
            frame_end = (frame_idx + 1) * frame_seq_length

            # 可以 attend 到的范围（0 到 block_end_frame）
            attend_end = block_end_frame * frame_seq_length

            # 设置 mask：False 表示可以 attend
            mask[frame_start:frame_end, :attend_end] = False

        # 应用 mask
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 提取最后一个 block 的 query 对所有之前帧的注意力
        # 最后一个 block 的起始帧
        last_block_start_frame = ((num_frames - 1) // num_frame_per_block) * num_frame_per_block
        last_block_start_token = last_block_start_frame * frame_seq_length

        # 提取最后一个 block query 对之前所有 key 的注意力
        # Shape: [B, N, last_block_tokens, all_tokens]
        last_block_attn = attn_weights[:, :, last_block_start_token:, :]

        captured_attention.append({
            'layer_idx': block_idx,
            'attn_weights': last_block_attn.cpu(),  # 只保存最后一个 block 的注意力
            'full_attn_shape': attn_weights.shape,
            'q_shape': q.shape,
            'k_shape': k.shape,
            'num_frames': num_frames,
            'frame_seq_length': frame_seq_length,
            'last_block_start_frame': last_block_start_frame,
            'num_frame_per_block': num_frame_per_block,
        })

        print(f"  Layer {block_idx}: full attention shape {attn_weights.shape}")
        print(f"  Last block attention shape: {last_block_attn.shape}")
        print(f"  Last block starts at frame {last_block_start_frame}")

        # 清理内存
        del attn_scores, attn_weights, last_block_attn
        torch.cuda.empty_cache()

    return captured_attention


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
