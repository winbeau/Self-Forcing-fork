"""Flash Attention 功能测试"""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.gpu
def test_flash_attention_basic():
    """测试 Flash Attention 基本功能"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except ImportError:
        pytest.skip("flash-attn not installed")

    print(f"\n✅ Flash Attention 版本: {flash_attn.__version__}")
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA 设备: {torch.cuda.get_device_name(0)}")

    # 准备测试数据 (必须是 CUDA + float16 或 bfloat16)
    batch_size = 2
    seq_len = 1024
    n_heads = 8
    head_dim = 64
    dtype = torch.float16
    device = "cuda"

    print("\n🚀 生成随机 Tensor...")
    q = torch.randn((batch_size, seq_len, n_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch_size, seq_len, n_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch_size, seq_len, n_heads, head_dim), device=device, dtype=dtype)

    print("⚡ 执行 Flash Attention 计算...")
    output = flash_attn_func(q, k, v)

    # 验证输出形状
    expected_shape = (batch_size, seq_len, n_heads, head_dim)
    assert output.shape == expected_shape, f"形状错误: {output.shape} != {expected_shape}"

    print(f"✓ Flash Attention 测试通过，输出形状: {output.shape}")
