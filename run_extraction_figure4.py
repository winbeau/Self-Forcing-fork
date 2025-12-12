#!/usr/bin/env python
"""
Figure 4 注意力权重提取脚本（修正版 v4）

关键修复：
- v3: 使用 pipeline.inference()，但只能捕获最后一个 block 的注意力（3帧 query）
- v4: 添加 --full_attention 模式，计算完整的 N帧×N帧 注意力矩阵

用法：
    # 完整注意力模式（推荐，21帧×21帧）
    python run_extraction_figure4.py \
        --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --output_path attention_cache_figure4.pt \
        --layer_indices 0 4 \
        --full_attention

    # KV cache 模式（只有最后3帧作为 query）
    python run_extraction_figure4.py \
        --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --output_path attention_cache_figure4.pt \
        --layer_indices 0 4
"""

import argparse
import torch
import os
import sys

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
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    # 加载配置
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    print("Initializing inference pipeline...")
    torch.set_grad_enabled(False)

    pipeline = CausalInferencePipeline(args=config, device=device)

    # 加载 checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    print(f"\nNum frames: {args.num_frames}")
    print(f"Layer indices: {args.layer_indices}")
    print(f"Prompt: {args.prompt}")
    print(f"Full attention mode: {args.full_attention}")

    # 创建输入
    batch_size = 1
    noise = torch.randn(
        [batch_size, args.num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    num_frame_per_block = config.get('num_frame_per_block', 3)

    # ========== 完整注意力模式（推荐用于 Figure 4 复现）==========
    if args.full_attention:
        print("\n" + "=" * 60)
        print("完整注意力模式：计算 N帧×N帧 注意力矩阵")
        print("提取论文 Figure 4 所需的子矩阵：")
        print(f"  Query: 最后 {num_frame_per_block} 帧 (帧 {args.num_frames - num_frame_per_block}-{args.num_frames - 1})")
        print(f"  Key: 前 {args.num_frames - num_frame_per_block} 帧 (帧 0-{args.num_frames - num_frame_per_block - 1})")
        print("=" * 60)

        # 先执行推理获取去噪后的 latent
        print("\n执行推理...")
        output = pipeline.inference(
            noise=noise,
            text_prompts=[args.prompt],
            return_latents=True,
        )
        # 处理返回值（可能是 tuple 或 tensor）
        if isinstance(output, tuple):
            output_latent = output[1]  # 第二个元素是 latent
            print(f"Output is tuple with {len(output)} elements, using latent")
        else:
            output_latent = output
        print(f"Output latent shape: {output_latent.shape}")

        # 计算完整注意力并提取 Figure 4 所需的子矩阵
        final_attentions = compute_attention_from_denoised(
            pipeline.generator, output_latent, args.layer_indices, num_frame_per_block, device
        )

        # 从返回的数据中获取 frame_seq_length
        frame_seq_length = final_attentions[0]['frame_seq_length'] if final_attentions else 1560
        num_key_frames = args.num_frames - num_frame_per_block

        # 保存
        save_data = {
            'attention_weights': [],
            'prompt': args.prompt,
            'num_frames': args.num_frames,
            'frame_seq_length': frame_seq_length,
            'num_frame_per_block': num_frame_per_block,
            'layer_indices': args.layer_indices,
            'is_logits': True,
            'capture_method': 'full_attention_figure4',
            'is_full_attention': True,
            # Figure 4 specific metadata
            'query_frames': list(range(args.num_frames - num_frame_per_block, args.num_frames)),
            'key_frames': list(range(num_key_frames)),
        }

        for attn in final_attentions:
            save_data['attention_weights'].append({
                'layer_idx': attn['layer_idx'],
                'attn_logits': attn['attn_logits'],  # 子矩阵：last chunk -> prev frames
                'is_logits': True,
                'num_frames': args.num_frames,
                'frame_seq_length': frame_seq_length,
                'num_heads': attn['attn_logits'].shape[1],
                'is_full_attention': True,
                # Figure 4 specific
                'query_start_frame': attn['query_start_frame'],
                'key_end_frame': attn['key_end_frame'],
                'num_query_frames': num_frame_per_block,
                'num_key_frames': num_key_frames,
                'last_block_start_frame': args.num_frames - num_frame_per_block,
            })

        torch.save(save_data, args.output_path)
        print(f"\nSaved to: {args.output_path}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("FIGURE 4 ATTENTION DATA SUMMARY")
        print("=" * 60)
        print(f"Query: frames {args.num_frames - num_frame_per_block}-{args.num_frames - 1} ({num_frame_per_block} frames)")
        print(f"Key: frames 0-{num_key_frames - 1} ({num_key_frames} frames)")
        print()
        for w in save_data['attention_weights']:
            layer = w['layer_idx']
            logits = w['attn_logits'].float()
            shape = tuple(logits.shape)
            q_frames = shape[2] // frame_seq_length
            k_frames = shape[3] // frame_seq_length
            print(f"Layer {layer}:")
            print(f"  Shape: {shape}")
            print(f"  Query tokens: {shape[2]} ({q_frames} frames × {frame_seq_length} tokens/frame)")
            print(f"  Key tokens: {shape[3]} ({k_frames} frames × {frame_seq_length} tokens/frame)")
            print(f"  Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

            # 计算帧间统计
            import numpy as np
            attn_np = logits[0, 0].numpy()  # 第一个 head
            avg_per_key = attn_np.mean(axis=0)  # [K]
            frame_means = [avg_per_key[f*frame_seq_length:(f+1)*frame_seq_length].mean()
                          for f in range(k_frames)]
            frame_means = np.array(frame_means)
            print(f"  Frame-wise attention (Head 1, first 5 frames): {frame_means[:5]}")
            print(f"  Frame-wise attention (Head 1, last 5 frames): {frame_means[-5:]}")
            diff_pct = (frame_means.max() - frame_means.min()) / np.abs(frame_means.mean()) * 100
            print(f"  Frame-wise diff: {diff_pct:.2f}%")

        return

    # ========== KV Cache 模式（原有逻辑）==========
    print("\n" + "=" * 60)
    print("KV Cache 模式：只捕获最后一个 block 的注意力")
    print("=" * 60)

    # 存储捕获的注意力
    captured_data = {
        'attentions': [],
        'layer_indices': args.layer_indices,
    }

    # Patch self_attn 的 forward 方法来捕获注意力
    original_forwards = {}

    def make_capture_forward(block_idx, original_forward):
        def capture_forward(self, *args, **kwargs):
            # 调用原始方法
            result = original_forward(*args, **kwargs)

            # 如果需要捕获这一层
            if block_idx in args_layer_indices and capture_enabled[0]:
                # 手动计算注意力 logits
                x = args[0] if len(args) > 0 else kwargs.get('x')
                if x is not None:
                    try:
                        b, s, d = x.shape
                        n = self.num_heads
                        head_dim = self.head_dim
                        scale = head_dim ** -0.5

                        q = self.norm_q(self.q(x))
                        k = self.norm_k(self.k(x))

                        q = q.view(b, s, n, head_dim).transpose(1, 2)
                        k = k.view(b, s, n, head_dim).transpose(1, 2)

                        attn_logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

                        captured_data['attentions'].append({
                            'layer_idx': block_idx,
                            'attn_logits': attn_logits.cpu().to(torch.float16),
                            'q_len': s,
                            'k_len': s,
                        })
                        print(f"  Captured layer {block_idx}: shape={attn_logits.shape}, "
                              f"range=[{attn_logits.min().item():.2f}, {attn_logits.max().item():.2f}]")
                    except Exception as e:
                        print(f"  Failed to capture layer {block_idx}: {e}")

            return result
        return capture_forward

    # 应用 patches
    args_layer_indices = args.layer_indices
    capture_enabled = [False]  # 用列表以便在闭包中修改

    wan_model = pipeline.generator.model
    for block_idx, block in enumerate(wan_model.blocks):
        if block_idx in args.layer_indices:
            original_forwards[block_idx] = block.self_attn.forward
            block.self_attn.forward = make_capture_forward(block_idx, block.self_attn.forward).__get__(
                block.self_attn, type(block.self_attn)
            )
            print(f"Patched layer {block_idx}")

    # 执行推理
    print("\n" + "=" * 60)
    print("执行完整推理流程...")
    print("=" * 60)

    # 在最后几个 block 时启用捕获
    num_frame_per_block = config.get('num_frame_per_block', 3)
    if config.independent_first_frame:
        total_blocks = 1 + (args.num_frames - 1) // num_frame_per_block
    else:
        total_blocks = args.num_frames // num_frame_per_block

    print(f"Total temporal blocks: {total_blocks}")
    print(f"Will capture on the last block")

    # 修改 pipeline 来在正确时机启用捕获
    original_inference = pipeline.inference
    block_counter = [0]

    def patched_inference(*args, **kwargs):
        # 直接调用，我们通过监控 generator 调用来判断时机
        return original_inference(*args, **kwargs)

    # 更简单的方法：直接在整个推理过程中捕获
    # 由于 KV cache 的存在，最后一个 block 的注意力会包含对之前所有帧的关注
    capture_enabled[0] = True

    try:
        output_latent = pipeline.inference(
            noise=noise,
            text_prompts=[args.prompt],
            return_latents=True,
        )
        print(f"Inference completed. Output shape: {output_latent.shape}")
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture_enabled[0] = False

    # 恢复原始方法
    for block_idx in original_forwards:
        wan_model.blocks[block_idx].self_attn.forward = original_forwards[block_idx]

    # 检查捕获结果
    if not captured_data['attentions']:
        print("\nNo attention captured via patch. Using fallback method...")
        captured_data['attentions'] = compute_attention_from_denoised(
            pipeline.generator, output_latent, args.layer_indices, num_frame_per_block, device
        )

    if not captured_data['attentions']:
        print("Error: No attention data captured!")
        sys.exit(1)

    # 筛选最后一个 block 的注意力（形状最大的那些）
    print(f"\nTotal captured: {len(captured_data['attentions'])} tensors")

    # 按层分组，每层取最后一个（对应最后的 temporal block）
    layer_attentions = {}
    for attn in captured_data['attentions']:
        layer_idx = attn['layer_idx']
        if layer_idx not in layer_attentions:
            layer_attentions[layer_idx] = []
        layer_attentions[layer_idx].append(attn)

    # 每层取最后一个
    final_attentions = []
    for layer_idx in sorted(layer_attentions.keys()):
        attns = layer_attentions[layer_idx]
        # 取 k_len 最大的（包含最多历史信息）
        attns_sorted = sorted(attns, key=lambda x: x['k_len'], reverse=True)
        final_attentions.append(attns_sorted[0])
        print(f"Layer {layer_idx}: selected shape {attns_sorted[0]['attn_logits'].shape}")

    # 保存
    frame_seq_length = 1560
    save_data = {
        'attention_weights': [],
        'prompt': args.prompt,
        'num_frames': args.num_frames,
        'frame_seq_length': frame_seq_length,
        'num_frame_per_block': num_frame_per_block,
        'layer_indices': args.layer_indices,
        'is_logits': True,
        'capture_method': 'monkey_patch_inference',
    }

    for attn in final_attentions:
        save_data['attention_weights'].append({
            'layer_idx': attn['layer_idx'],
            'attn_logits': attn['attn_logits'],
            'is_logits': True,
            'num_frames': args.num_frames,
            'frame_seq_length': frame_seq_length,
            'last_block_start_frame': args.num_frames - num_frame_per_block,
            'num_frame_per_block': num_frame_per_block,
            'num_heads': attn['attn_logits'].shape[1],
        })

    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("ATTENTION DATA SUMMARY")
    print("=" * 60)
    for w in save_data['attention_weights']:
        layer = w['layer_idx']
        logits = w['attn_logits'].float()
        shape = tuple(logits.shape)
        print(f"Layer {layer}: shape={shape}")
        print(f"  Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

        # 计算帧间差异
        if len(shape) == 4:
            B, H, Q, K = shape
            num_key_frames = K // frame_seq_length
            if num_key_frames > 1:
                attn_np = logits[0, 0].numpy()  # 第一个 head
                avg_per_key = attn_np.mean(axis=0)
                frame_means = [avg_per_key[f*frame_seq_length:(f+1)*frame_seq_length].mean()
                              for f in range(num_key_frames)]
                import numpy as np
                frame_means = np.array(frame_means)
                diff_pct = (frame_means.max() - frame_means.min()) / np.abs(frame_means.mean()) * 100
                print(f"  Frame-wise diff: {diff_pct:.2f}%")
                print(f"  Frame means: {frame_means[:3]}... {frame_means[-3:]}")


def compute_attention_from_denoised(generator, denoised_latent, layer_indices, num_frame_per_block, device):
    """
    使用去噪后的 latent 计算 Figure 4 所需的注意力矩阵。

    优化：只计算论文需要的子矩阵，避免 OOM：
    - Query: 最后 3 帧 (帧 18-20)
    - Key: 前 18 帧 (帧 0-17)

    这样只需计算 4680 × 28080 的矩阵，而不是 32760 × 32760。
    """
    import gc
    print("\n计算 Figure 4 所需的注意力子矩阵（优化显存）...")

    wan_model = generator.model
    batch_size, num_frames = denoised_latent.shape[:2]

    # Get model dtype from patch_embedding weights
    model_dtype = wan_model.patch_embedding.weight.dtype

    # [B, T, C, H, W] -> [B, C, T, H, W]
    x = denoised_latent.permute(0, 2, 1, 3, 4)
    x = x.to(dtype=model_dtype)
    x = wan_model.patch_embedding(x)
    x = x.flatten(2).transpose(1, 2)

    seq_len = x.shape[1]
    frame_seq_length = seq_len // num_frames

    print(f"Embedded shape: {x.shape}")
    print(f"Total frames: {num_frames}, Tokens/frame: {frame_seq_length}")

    # 计算关键位置
    num_query_frames = num_frame_per_block  # 3
    num_key_frames = num_frames - num_frame_per_block  # 18

    query_start_token = num_key_frames * frame_seq_length  # 18 * 1560 = 28080
    key_end_token = query_start_token  # 前18帧的结束位置

    print(f"\nFigure 4 子矩阵计算：")
    print(f"  Query: 帧 {num_key_frames}-{num_frames-1} ({num_query_frames} 帧, tokens {query_start_token}-{seq_len-1})")
    print(f"  Key: 帧 0-{num_key_frames-1} ({num_key_frames} 帧, tokens 0-{key_end_token-1})")
    print(f"  子矩阵大小: {num_query_frames * frame_seq_length} × {num_key_frames * frame_seq_length}")
    estimated_mem = (num_query_frames * frame_seq_length * num_key_frames * frame_seq_length * 4 * 12) / (1024**3)
    print(f"  预计显存: ~{estimated_mem:.2f} GB (12 heads × float32)")

    captured = []

    for block_idx, block in enumerate(wan_model.blocks):
        if block_idx not in layer_indices:
            continue

        print(f"\nProcessing layer {block_idx}...")
        torch.cuda.empty_cache()
        gc.collect()

        self_attn = block.self_attn
        x_norm = block.norm1(x)

        b, s, d = x_norm.shape
        n = self_attn.num_heads
        head_dim = self_attn.head_dim
        scale = head_dim ** -0.5

        # 计算 Q 和 K（完整序列）
        q_full = self_attn.norm_q(self_attn.q(x_norm))
        k_full = self_attn.norm_k(self_attn.k(x_norm))

        # 只提取需要的部分
        # Query: 最后 3 帧
        q = q_full[:, query_start_token:, :]  # [B, 4680, D]
        # Key: 前 18 帧
        k = k_full[:, :key_end_token, :]  # [B, 28080, D]

        del q_full, k_full
        torch.cuda.empty_cache()

        q = q.view(b, -1, n, head_dim).transpose(1, 2)  # [B, N, 4680, D]
        k = k.view(b, -1, n, head_dim).transpose(1, 2)  # [B, N, 28080, D]

        print(f"  Q shape: {q.shape}, K shape: {k.shape}")

        # 计算注意力 logits: [B, N, 4680, 28080]
        attn_logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

        print(f"  Attention logits shape: {attn_logits.shape}")
        print(f"  Range: [{attn_logits.min().item():.4f}, {attn_logits.max().item():.4f}]")

        captured.append({
            'layer_idx': block_idx,
            'attn_logits': attn_logits.cpu().to(torch.float16),
            'q_len': attn_logits.shape[2],
            'k_len': attn_logits.shape[3],
            'num_frames': num_frames,
            'frame_seq_length': frame_seq_length,
            'num_frame_per_block': num_frame_per_block,
            'query_start_frame': num_key_frames,
            'key_end_frame': num_key_frames,
            'num_query_frames': num_query_frames,
            'num_key_frames': num_key_frames,
        })

        del q, k, attn_logits
        torch.cuda.empty_cache()
        gc.collect()

    return captured


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="attention_cache_figure4.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--layer_indices", type=int, nargs='+', default=[0, 4])
    parser.add_argument("--full_attention", action="store_true", default=False,
                        help="计算完整的 N帧×N帧 注意力矩阵（推荐）")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
