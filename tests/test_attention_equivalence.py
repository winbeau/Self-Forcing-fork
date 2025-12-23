#!/usr/bin/env python
"""
验证 attention_with_weights 的数学语义与参考实现/快路径一致。

说明：
- “完全一样”通常只能做到语义一致 + 数值近似（不同 kernel/fused 实现会带来浮点误差）。
- 本测试优先用 PyTorch SDPA + 显式 mask 作为参考；如果安装了 flash-attn，再额外对比快路径。
"""

import os
import sys

import pytest
import torch
import importlib.util
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_ATTN_MOD = None


def _load_attention_module():
    global _ATTN_MOD
    if _ATTN_MOD is not None:
        return _ATTN_MOD
    # 避免 import `wan` 包触发 CUDA/torch.compile 的副作用：直接从文件加载 attention.py。
    path = Path(__file__).resolve().parents[1] / "wan" / "modules" / "attention.py"
    spec = importlib.util.spec_from_file_location("_wan_modules_attention", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ATTN_MOD = mod
    return mod


def _repeat_kv_for_gqa(q, k, v):
    n_q, n_k = q.shape[2], k.shape[2]
    if n_q == n_k:
        return k, v
    if n_q % n_k != 0:
        raise ValueError(f"Nq must be divisible by Nk, got Nq={n_q}, Nk={n_k}")
    repeat_factor = n_q // n_k
    k = k.repeat_interleave(repeat_factor, dim=2)
    v = v.repeat_interleave(repeat_factor, dim=2)
    return k, v


def _build_allow_mask(*, lq, lk, device, q_lens=None, k_lens=None, causal=False, window_size=(-1, -1)):
    # SDPA 的 bool mask 语义：True 表示“允许”，False 表示“屏蔽”。
    allow = torch.ones((1, 1, lq, lk), dtype=torch.bool, device=device)

    if k_lens is not None:
        key_pos = torch.arange(lk, device=device).view(1, 1, 1, lk)
        allow = allow & (key_pos < k_lens.view(-1, 1, 1, 1))

    if q_lens is not None:
        q_pos = torch.arange(lq, device=device).view(1, 1, lq, 1)
        allow = allow & (q_pos < q_lens.view(-1, 1, 1, 1))

    # 对齐非方阵 Q/K：query i 对应 key i + (lk - lq)
    offset = lk - lq
    q_pos_2d = torch.arange(lq, device=device).view(lq, 1)
    k_pos_2d = torch.arange(lk, device=device).view(1, lk)
    center = q_pos_2d + offset

    if causal:
        allow = allow & (k_pos_2d <= center).view(1, 1, lq, lk)

    if window_size != (-1, -1):
        left, right = window_size
        if left < 0:
            left = lk
        if right < 0:
            right = lk
        allow = allow & ((k_pos_2d >= (center - left)) & (k_pos_2d <= (center + right))).view(1, 1, lq, lk)

    return allow


def _sdpa_reference(q, k, v, *, allow_mask, softmax_scale=None, q_scale=None, dropout_p=0.0):
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)

    if q_scale is not None:
        q_ = q_ * q_scale
    if softmax_scale is not None:
        q_ = q_ * softmax_scale

    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=allow_mask, is_causal=False, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "case",
    [
        # 方阵：baseline
        dict(lq=64, lk=64, n_q=4, n_k=4, causal=False, window_size=(-1, -1), k_lens=None),
        dict(lq=64, lk=64, n_q=4, n_k=4, causal=True, window_size=(-1, -1), k_lens=None),
        # KV cache 常见非方阵：Lk > Lq，重点覆盖 causal offset 语义
        dict(lq=32, lk=96, n_q=4, n_k=4, causal=True, window_size=(-1, -1), k_lens=None),
        # 局部注意力：窗口 mask
        dict(lq=64, lk=64, n_q=4, n_k=4, causal=False, window_size=(8, 8), k_lens=None),
        dict(lq=64, lk=64, n_q=4, n_k=4, causal=True, window_size=(16, 0), k_lens=None),
        # k_lens padding mask：batch 内不同有效长度
        dict(lq=64, lk=64, n_q=4, n_k=4, causal=False, window_size=(-1, -1), k_lens=torch.tensor([64, 48])),
        # GQA/MQA：Nq 是 Nk 的倍数
        dict(lq=64, lk=64, n_q=8, n_k=2, causal=False, window_size=(-1, -1), k_lens=None),
    ],
    ids=[
        "square-norm",
        "square-causal",
        "kv-cache-causal-offset",
        "window-bidirectional",
        "window-causal-left",
        "k-lens-mask",
        "gqa-repeat-kv",
    ],
)
def test_attention_with_weights_matches_sdpa(case):
    attention_with_weights = _load_attention_module().attention_with_weights

    torch.manual_seed(0)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    b = 2
    lq, lk = case["lq"], case["lk"]
    n_q, n_k = case["n_q"], case["n_k"]
    d = 64

    q = torch.randn((b, lq, n_q, d), device=device, dtype=dtype)
    k = torch.randn((b, lk, n_k, d), device=device, dtype=dtype)
    v = torch.randn((b, lk, n_k, d), device=device, dtype=dtype)

    k, v = _repeat_kv_for_gqa(q, k, v)

    k_lens = case["k_lens"]
    if k_lens is not None:
        k_lens = k_lens.to(device=device)

    allow = _build_allow_mask(
        lq=lq,
        lk=lk,
        device=device,
        q_lens=None,
        k_lens=k_lens,
        causal=case["causal"],
        window_size=case["window_size"],
    )
    if allow.shape[0] == 1 and b != 1:
        allow = allow.expand(b, -1, -1, -1).contiguous()

    out_with, _scores = attention_with_weights(
        q=q,
        k=k,
        v=v,
        k_lens=k_lens,
        dropout_p=0.0,
        causal=case["causal"],
        window_size=case["window_size"],
        dtype=dtype,
        return_logits=True,
    )
    out_sdpa = _sdpa_reference(q, k, v, allow_mask=allow, dropout_p=0.0)

    tol = dict(rtol=2e-2, atol=2e-2) if dtype in (torch.float16, torch.bfloat16) else dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_with, out_sdpa, **tol)


@pytest.mark.parametrize(
    "case",
    [
        dict(lq=64, lk=64, causal=False, window_size=(-1, -1), k_lens=None),
        dict(lq=64, lk=64, causal=True, window_size=(-1, -1), k_lens=None),
        dict(lq=32, lk=96, causal=True, window_size=(-1, -1), k_lens=None),
        dict(lq=64, lk=64, causal=False, window_size=(8, 8), k_lens=torch.tensor([64, 48])),
    ],
    ids=["square-norm", "square-causal", "kv-cache-causal-offset", "window-plus-kmask"],
)
def test_attention_with_weights_matches_flash_attention(case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mod = _load_attention_module()
    FLASH_ATTN_2_AVAILABLE = mod.FLASH_ATTN_2_AVAILABLE
    FLASH_ATTN_3_AVAILABLE = mod.FLASH_ATTN_3_AVAILABLE
    attention_with_weights = mod.attention_with_weights
    flash_attention = mod.flash_attention

    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        pytest.skip("flash-attn not available")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    b, d, n = 2, 64, 4
    lq, lk = case["lq"], case["lk"]

    q = torch.randn((b, lq, n, d), device=device, dtype=dtype)
    k = torch.randn((b, lk, n, d), device=device, dtype=dtype)
    v = torch.randn((b, lk, n, d), device=device, dtype=dtype)

    k_lens = case["k_lens"]
    if k_lens is not None:
        k_lens = k_lens.to(device=device)

    out_flash = flash_attention(
        q=q,
        k=k,
        v=v,
        k_lens=k_lens,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=case["causal"],
        window_size=case["window_size"],
        dtype=dtype,
    )
    out_with, _scores = attention_with_weights(
        q=q,
        k=k,
        v=v,
        k_lens=k_lens,
        dropout_p=0.0,
        causal=case["causal"],
        window_size=case["window_size"],
        dtype=dtype,
        return_logits=True,
    )

    torch.testing.assert_close(out_with, out_flash, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
