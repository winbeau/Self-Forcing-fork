#!/usr/bin/env python
"""
Attention Weight Extraction Script for Figure 4 Reproduction

This script runs a short inference to extract temporal self-attention weights
from the Self-Forcing model. The weights are saved to disk for visualization.

Usage:
    python run_extraction.py \
        --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --output_path attention_cache.pt \
        --prompt "A beautiful sunset over the ocean" \
        --num_frames 21 \
        --layer_indices 15  # mid-layer, or omit for all layers

Requirements:
    - Model checkpoint downloaded
    - GPU with sufficient memory (recommend >= 24GB)
"""

import argparse
import torch
import os
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE
from utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Extract attention weights for Figure 4")
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml",
                        help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt",
                        help="Path to the checkpoint file")
    parser.add_argument("--output_path", type=str, default="attention_cache.pt",
                        help="Path to save the attention weights")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting, detailed feathers, slow motion",
                        help="Text prompt for video generation")
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Number of frames to generate")
    parser.add_argument("--layer_indices", type=int, nargs='+', default=None,
                        help="Specific layer indices to capture (default: all layers)")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--capture_block_idx", type=int, default=None,
                        help="Only capture attention for a specific generation block (0-indexed)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Check for GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load config
    print(f"\nLoading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # Initialize pipeline
    print("Initializing pipeline...")
    torch.set_grad_enabled(False)
    pipeline = CausalInferencePipeline(config, device=device)

    # Load checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            print(f"Warning: '{key}' not found in checkpoint, trying 'generator'")
            pipeline.generator.load_state_dict(state_dict['generator'])
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}, using random weights")

    # Move to GPU and set dtype
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    print(f"\nModel has {pipeline.num_transformer_blocks} transformer blocks")
    print(f"Frame sequence length: {pipeline.frame_seq_length}")
    print(f"Frames per block: {pipeline.num_frame_per_block}")

    # Prepare input
    print(f"\nGenerating video with prompt: '{args.prompt}'")
    print(f"Number of frames: {args.num_frames}")

    # Create noise
    batch_size = 1
    sampled_noise = torch.randn(
        [batch_size, args.num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # Configure attention weight capture
    layer_indices = args.layer_indices
    if layer_indices is not None:
        print(f"Capturing attention from layers: {layer_indices}")
    else:
        # Default to middle layer for Figure 4
        mid_layer = pipeline.num_transformer_blocks // 2
        layer_indices = [mid_layer]
        print(f"Capturing attention from mid-layer: {layer_indices}")

    # Enable attention capture
    ATTENTION_WEIGHT_CAPTURE.enable(layer_indices=layer_indices)

    # Run inference
    print("\nRunning inference...")
    try:
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[args.prompt],
            return_latents=False,
        )
        print(f"Generated video shape: {video.shape}")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        # Always save what we captured
        pass

    # Save captured attention weights
    if len(ATTENTION_WEIGHT_CAPTURE.captured_weights) > 0:
        print(f"\nCaptured {len(ATTENTION_WEIGHT_CAPTURE.captured_weights)} attention weight tensors")

        # Prepare save data
        save_data = {
            'attention_weights': ATTENTION_WEIGHT_CAPTURE.captured_weights,
            'layer_indices': layer_indices,
            'prompt': args.prompt,
            'num_frames': args.num_frames,
            'frame_seq_length': pipeline.frame_seq_length,
            'num_frame_per_block': pipeline.num_frame_per_block,
            'num_transformer_blocks': pipeline.num_transformer_blocks,
            'config': {
                'denoising_step_list': config.get('denoising_step_list', None),
            }
        }

        # Print some statistics about captured weights
        for i, w in enumerate(save_data['attention_weights']):
            print(f"  Weight {i}: layer={w['layer_idx']}, "
                  f"attn_shape={w['attn_weights'].shape}, "
                  f"q_shape={w['q_shape']}, k_shape={w['k_shape']}")

        # Save to disk
        torch.save(save_data, args.output_path)
        print(f"\nSaved attention weights to: {args.output_path}")
    else:
        print("\nWarning: No attention weights were captured!")
        print("This might happen if:")
        print("  - The specified layer indices don't match any layers")
        print("  - The inference didn't use the attention function")

    # Disable capture
    ATTENTION_WEIGHT_CAPTURE.disable()

    print("\nDone!")


if __name__ == "__main__":
    main()
