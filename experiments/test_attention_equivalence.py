#!/usr/bin/env python
"""
验证 attention_with_weights 的数学逻辑与快路径一致。

注意：FlashAttention / SDPA 与显式 matmul-softmax 的实现会有数值误差（尤其是 bf16）。
本测试使用相对宽松的 atol/rtol 来验证“近似一致”。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sdpa_reference(q, k, v, *, dropout_p=0.0, softmax_scale=None, causal=False):
    # attention_with_weights 内部会把输入从 [B, L, H, D] 转成 [B, H, L, D] 再算。
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)

    if softmax_scale is not None:
        q_ = q_ * softmax_scale
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, is_causal=causal, dropout_p=dropout_p
        )
    else:
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, is_causal=causal, dropout_p=dropout_p
        )

    return attn_out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("causal", [False, True])
def test_attention_with_weights_matches_sdpa(causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from wan.modules.attention import attention_with_weights

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    b, lq, lk, h, d = 2, 64, 64, 4, 64
    dtype = torch.bfloat16

    q = torch.randn((b, lq, h, d), device=device, dtype=dtype)
    k = torch.randn((b, lk, h, d), device=device, dtype=dtype)
    v = torch.randn((b, lk, h, d), device=device, dtype=dtype)

    out_with, _scores = attention_with_weights(
        q=q, k=k, v=v, dropout_p=0.0, causal=causal, dtype=dtype, return_logits=True
    )
    out_sdpa = _sdpa_reference(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)

    torch.testing.assert_close(out_with, out_sdpa, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_with_weights_matches_flash_attention(causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from wan.modules.attention import (
        FLASH_ATTN_2_AVAILABLE,
        FLASH_ATTN_3_AVAILABLE,
        attention_with_weights,
        flash_attention,
    )

    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        pytest.skip("flash-attn not available")

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    b, lq, lk, h, d = 2, 64, 64, 4, 64
    dtype = torch.bfloat16

    q = torch.randn((b, lq, h, d), device=device, dtype=dtype)
    k = torch.randn((b, lk, h, d), device=device, dtype=dtype)
    v = torch.randn((b, lk, h, d), device=device, dtype=dtype)

    out_flash = flash_attention(
        q=q,
        k=k,
        v=v,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=causal,
        dtype=dtype,
    )
    out_with, _scores = attention_with_weights(
        q=q, k=k, v=v, dropout_p=0.0, causal=causal, dtype=dtype, return_logits=True
    )

    torch.testing.assert_close(out_with, out_flash, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short"]))

