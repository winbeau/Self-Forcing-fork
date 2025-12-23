#!/usr/bin/env python
"""
测试 ATTENTION_WEIGHT_CAPTURE 机制是否正常工作。

验证：
1. 模块化索引正确工作
2. 能捕获到多个 block/timestep 的 attention
3. 最后一个 block 的 K 包含所有历史帧

注意：GPU 测试需要 --run-slow 选项
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAttentionCaptureUnit:
    """单元测试（不需要 GPU）"""

    def test_modular_index(self):
        """测试模块化索引正确工作"""
        from wan.modules.attention import AttentionWeightCapture

        capture = AttentionWeightCapture()
        capture.enable(layer_indices=[0, 4], num_layers=30)

        # 模拟多个 step 的调用
        results = []
        for step in range(3):  # 3个 step
            for layer in range(30):  # 30层
                capture.current_layer_idx = step * 30 + layer
                if capture.should_capture():
                    results.append((step, layer, capture.get_effective_layer_idx()))

        # 应该每个 step 都捕获 layer 0 和 4
        assert len(results) == 6, f"Expected 6 captures, got {len(results)}"

        # 验证捕获的是正确的层
        for step, layer, effective in results:
            assert effective in [0, 4], f"Unexpected effective layer: {effective}"
            assert layer == effective, f"Layer mismatch: {layer} vs {effective}"

        print(f"✓ 模块化索引测试通过: 捕获了 {len(results)} 个 attention")
        print(f"  捕获详情: {results}")


@pytest.mark.slow
@pytest.mark.gpu
class TestAttentionCaptureGPU:
    """GPU 集成测试（需要 --run-slow）"""

    def test_capture_shape_grows_with_kv_cache(self):
        """测试 KV cache 模式下，K 的长度随着 block 增加而增长"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from omegaconf import OmegaConf
        from pipeline.causal_inference import CausalInferencePipeline
        from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE
        from utils.misc import set_seed

        print("\n📁 加载配置...")
        set_seed(42)
        device = torch.device("cuda:0")

        # 加载配置
        config = OmegaConf.load("configs/self_forcing_dmd.yaml")
        default_config = OmegaConf.load("configs/default_config.yaml")
        config = OmegaConf.merge(default_config, config)

        torch.set_grad_enabled(False)
        pipeline = CausalInferencePipeline(args=config, device=device)
        pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
        pipeline.eval()

        num_layers = len(pipeline.generator.model.blocks)
        num_frames = 21
        num_frame_per_block = config.get('num_frame_per_block', 3)

        noise = torch.randn(
            [1, num_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
        )

        # 只捕获 layer 0
        ATTENTION_WEIGHT_CAPTURE.enable(
            layer_indices=[0],
            capture_logits=True,
            num_layers=num_layers
        )

        try:
            pipeline.inference(
                noise=noise,
                text_prompts=["A test video"],
                return_latents=True,
            )
            captured = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
        finally:
            ATTENTION_WEIGHT_CAPTURE.disable()

        print(f"\n捕获到 {len(captured)} 个 attention 矩阵")

        # 验证捕获了多个 attention
        # 注意：实际捕获数量可能因为 cross attention 等原因而更多
        assert len(captured) > 0, "Should capture some attention"

        # 分析 K 的长度变化
        k_lengths = [c['k_shape'][1] for c in captured]
        print(f"K 长度序列（前12个）: {k_lengths[:12]}")

        # K 长度应该随着 block 增加而增长
        frame_seq_length = 60 * 104 // 4  # 1560

        # 检查 K 长度的最大值（最后一个 block）
        max_k_len = max(k_lengths)
        expected_k_len = num_frames * frame_seq_length  # 所有帧

        print(f"最大 K 长度: {max_k_len}")
        print(f"期望的 K 长度（包含所有帧）: {expected_k_len}")

        # 验证最后一个 block 的 K 包含所有历史帧
        min_k_len = (num_frames - num_frame_per_block) * frame_seq_length
        assert max_k_len >= min_k_len, \
            f"Max K length {max_k_len} is less than expected minimum {min_k_len}"

        print(f"✓ KV cache 测试通过: K 长度最大为 {max_k_len // frame_seq_length} 帧")

    def test_last_block_has_full_history(self):
        """测试最后一个 block 的 attention 包含完整历史"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from omegaconf import OmegaConf
        from pipeline.causal_inference import CausalInferencePipeline
        from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE
        from utils.misc import set_seed

        set_seed(42)
        device = torch.device("cuda:0")

        config = OmegaConf.load("configs/self_forcing_dmd.yaml")
        default_config = OmegaConf.load("configs/default_config.yaml")
        config = OmegaConf.merge(default_config, config)

        torch.set_grad_enabled(False)
        pipeline = CausalInferencePipeline(args=config, device=device)
        pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
        pipeline.eval()

        num_layers = len(pipeline.generator.model.blocks)
        num_frames = 21
        num_frame_per_block = 3
        frame_seq_length = 1560

        noise = torch.randn(
            [1, num_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
        )

        ATTENTION_WEIGHT_CAPTURE.enable(
            layer_indices=[0, 4],
            capture_logits=True,
            num_layers=num_layers
        )

        try:
            pipeline.inference(
                noise=noise,
                text_prompts=["A test video"],
                return_latents=True,
            )
            captured = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
        finally:
            ATTENTION_WEIGHT_CAPTURE.disable()

        # 按层分组，取 K 最大的（最后一个 block）
        layer_attentions = {}
        for attn in captured:
            layer_idx = attn['layer_idx']
            if layer_idx not in layer_attentions:
                layer_attentions[layer_idx] = []
            layer_attentions[layer_idx].append(attn)

        for layer_idx in [0, 4]:
            attns = layer_attentions[layer_idx]
            # 按 K 长度排序
            attns_sorted = sorted(attns, key=lambda x: x['k_shape'][1], reverse=True)
            best = attns_sorted[0]

            q_len = best['q_shape'][1]
            k_len = best['k_shape'][1]

            # Q 应该是 3 帧
            expected_q = num_frame_per_block * frame_seq_length
            assert q_len == expected_q, f"Layer {layer_idx}: Q length {q_len} != expected {expected_q}"

            # K 应该至少是 18 帧（所有历史帧）
            min_k = (num_frames - num_frame_per_block) * frame_seq_length
            assert k_len >= min_k, f"Layer {layer_idx}: K length {k_len} < expected minimum {min_k}"

            num_q_frames = q_len // frame_seq_length
            num_k_frames = k_len // frame_seq_length

            print(f"Layer {layer_idx}: Q={num_q_frames}帧, K={num_k_frames}帧")

        print(f"✓ 最后一个 block 包含完整历史测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
