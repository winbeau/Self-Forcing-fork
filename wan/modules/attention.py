# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import gc

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'attention_with_weights',
    'ATTENTION_WEIGHT_CAPTURE',
    'STREAMING_FRAME_ATTENTION_CAPTURE',
]


# ============================================================
# Streaming Frame-Level Attention Capture (Memory Efficient)
# ============================================================

class StreamingFrameAttentionCapture:
    """
    流式帧级注意力捕获器（内存高效版本）。

    核心思想：
    1. 使用 flash attention 进行实际的前向传播（高效，无 OOM）
    2. 分块计算 Q @ K^T 得到 attention logits
    3. 立即聚合到 frame-level 并释放 token-level 数据

    内存使用：O(Q_tokens × chunk_K_tokens) 而非 O(Q_tokens × all_K_tokens)
    """

    def __init__(self):
        self.enabled = False
        self.layer_indices = None
        self.current_layer_idx = 0
        self.num_layers = 60  # 30 blocks × 2 (self + cross)

        # Frame-level 配置
        self.frame_seq_length = 1560
        self.num_heads = 12
        self.chunk_frames = 3  # 每次处理的 K frames 数量

        # 回调函数：接收 frame-level attention 数据
        self.on_frame_attention = None

    def enable(
        self,
        layer_indices=None,
        num_layers=60,
        frame_seq_length=1560,
        num_heads=12,
        chunk_frames=3,
        on_frame_attention=None,
    ):
        """
        启用流式帧级注意力捕获。

        Args:
            layer_indices: 要捕获的层索引列表（self-attn 索引）
            num_layers: 总层数（用于取模）
            frame_seq_length: 每帧的 token 数量
            num_heads: 注意力头数
            chunk_frames: 每次处理的 K frames 数量（控制内存）
            on_frame_attention: 回调函数，签名 (layer_idx, frame_attn, q_frames, k_frames)
                frame_attn: [num_heads, q_frames, k_frames] 的 frame-level attention logits
        """
        self.enabled = True
        self.layer_indices = layer_indices
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.chunk_frames = chunk_frames
        self.on_frame_attention = on_frame_attention
        self.current_layer_idx = 0

    def disable(self):
        """禁用捕获。"""
        self.enabled = False
        self.current_layer_idx = 0
        self.on_frame_attention = None

    def should_capture(self):
        """检查是否应该捕获当前层。"""
        if not self.enabled:
            return False
        if self.layer_indices is None:
            return True
        effective_idx = self.current_layer_idx % self.num_layers
        return effective_idx in self.layer_indices

    def get_effective_layer_idx(self):
        """获取当前的有效层索引（模块化后）。"""
        return self.current_layer_idx % self.num_layers

    def compute_frame_attention_chunked(self, q, k, softmax_scale=None):
        """
        分块计算 frame-level attention logits。

        Args:
            q: Query 张量 [B, N, Lq, C]（已转置）
            k: Key 张量 [B, N, Lk, C]（已转置）
            softmax_scale: 缩放因子

        Returns:
            frame_attn: [N, q_frames, k_frames] 的 frame-level attention logits
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        b, n, lq, c = q.shape
        lk = k.shape[2]

        q_frames = lq // self.frame_seq_length
        k_frames = lk // self.frame_seq_length

        # 输出：frame-level attention [num_heads, q_frames, k_frames]
        frame_attn = torch.zeros(n, q_frames, k_frames, dtype=torch.float32, device='cpu')

        # 分块处理 K frames
        for kf_start in range(0, k_frames, self.chunk_frames):
            kf_end = min(kf_start + self.chunk_frames, k_frames)
            k_token_start = kf_start * self.frame_seq_length
            k_token_end = kf_end * self.frame_seq_length

            # 取出 K chunk: [B, N, chunk_tokens, C]
            k_chunk = k[:, :, k_token_start:k_token_end, :]

            # 对每个 Q frame 计算 attention
            for qf in range(q_frames):
                q_token_start = qf * self.frame_seq_length
                q_token_end = (qf + 1) * self.frame_seq_length

                # 取出 Q frame: [B, N, frame_tokens, C]
                q_frame = q[:, :, q_token_start:q_token_end, :]

                # 计算 attention scores: [B, N, frame_tokens, chunk_tokens]
                scores = torch.matmul(q_frame.float(), k_chunk.float().transpose(-2, -1)) * softmax_scale

                # 聚合到 frame-level（对每个 K frame 分别计算）
                for kf_local in range(kf_end - kf_start):
                    kf = kf_start + kf_local
                    k_local_start = kf_local * self.frame_seq_length
                    k_local_end = (kf_local + 1) * self.frame_seq_length

                    # 取出这个 K frame 对应的 scores
                    frame_scores = scores[:, :, :, k_local_start:k_local_end]  # [B, N, 1560, 1560]

                    # 计算均值：[N]
                    frame_attn[:, qf, kf] = frame_scores.mean(dim=(0, 2, 3)).cpu()

                # 释放中间结果
                del scores

            # 释放 K chunk
            del k_chunk
            torch.cuda.empty_cache()

        return frame_attn

    def capture_and_forward(self, q, k, v, flash_attn_fn, **kwargs):
        """
        捕获 frame-level attention 并执行 flash attention 前向传播。

        Args:
            q, k, v: 输入张量 [B, L, N, C]
            flash_attn_fn: Flash attention 函数
            **kwargs: 传递给 flash_attn_fn 的其他参数

        Returns:
            out: Flash attention 的输出
        """
        # 1. 执行 flash attention 得到输出
        out = flash_attn_fn(q=q, k=k, v=v, **kwargs)

        # 2. 如果需要捕获，分块计算 frame-level attention
        if self.should_capture():
            # 转置为 [B, N, L, C] 格式
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)

            # 处理 q_scale
            q_scale = kwargs.get('q_scale')
            if q_scale is not None:
                q_t = q_t * q_scale

            # 计算 frame-level attention
            softmax_scale = kwargs.get('softmax_scale')
            frame_attn = self.compute_frame_attention_chunked(q_t, k_t, softmax_scale)

            # 调用回调
            if self.on_frame_attention is not None:
                q_frames = q.shape[1] // self.frame_seq_length
                k_frames = k.shape[1] // self.frame_seq_length
                self.on_frame_attention(
                    layer_idx=self.get_effective_layer_idx(),
                    frame_attn=frame_attn,
                    q_frames=q_frames,
                    k_frames=k_frames,
                )

            # 清理
            del q_t, k_t, frame_attn
            gc.collect()
            torch.cuda.empty_cache()

        self.current_layer_idx += 1
        return out


# 全局实例
STREAMING_FRAME_ATTENTION_CAPTURE = StreamingFrameAttentionCapture()


# Global attention weight capture configuration
class AttentionWeightCapture:
    """
    全局注意力权重捕获配置。

    重要：捕获的是 pre-softmax logits（注意力分数），而非 softmax 后的概率！
    这对于复现论文 Figure 4 至关重要，因为论文中的 Y 轴范围是 [-4, 6]。
    """
    def __init__(self):
        self.enabled = False
        self.layer_indices = None  # None 表示捕获所有层，或者是层索引列表
        self.captured_weights = []  # 捕获的注意力权重列表
        self.current_layer_idx = 0  # 当前前向传播的层索引
        self.capture_logits = True  # 是否捕获 pre-softmax logits（默认 True）
        self.num_layers = 30  # Wan 模型的层数，用于取模

    def enable(self, layer_indices=None, capture_logits=True, num_layers=30):
        """
        启用注意力权重捕获。

        Args:
            layer_indices: 要捕获的层索引列表，None 表示全部
            capture_logits: 如果 True，捕获 pre-softmax logits；否则捕获 post-softmax probs
            num_layers: 模型的总层数，用于 current_layer_idx 取模
        """
        self.enabled = True
        self.layer_indices = layer_indices
        self.capture_logits = capture_logits
        self.num_layers = num_layers
        self.captured_weights = []
        self.current_layer_idx = 0

    def disable(self):
        """禁用注意力权重捕获。"""
        self.enabled = False
        self.captured_weights = []
        self.current_layer_idx = 0

    def reset(self):
        """重置捕获的权重，用于新的前向传播。"""
        self.captured_weights = []
        self.current_layer_idx = 0

    def should_capture(self):
        """检查是否应该捕获当前层（使用模块化索引）。"""
        if not self.enabled:
            return False
        if self.layer_indices is None:
            return True
        # 使用模块化索引，这样每个 denoising step 的相同层都会被检查
        effective_layer_idx = self.current_layer_idx % self.num_layers
        return effective_layer_idx in self.layer_indices

    def get_effective_layer_idx(self):
        """获取当前的有效层索引（模块化后）。"""
        return self.current_layer_idx % self.num_layers

    def save(self, path):
        """保存捕获的权重到磁盘。"""
        import torch
        torch.save({
            'attention_weights': self.captured_weights,
            'layer_indices': self.layer_indices,
            'capture_logits': self.capture_logits,
        }, path)
        print(f"Saved attention weights to {path}")


ATTENTION_WEIGHT_CAPTURE = AttentionWeightCapture()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # 优先检查流式帧级捕获（内存高效）
    if STREAMING_FRAME_ATTENTION_CAPTURE.enabled:
        if STREAMING_FRAME_ATTENTION_CAPTURE.should_capture():
            # 使用流式捕获：flash attention + 分块计算 frame-level attention
            out = STREAMING_FRAME_ATTENTION_CAPTURE.capture_and_forward(
                q=q, k=k, v=v,
                flash_attn_fn=flash_attention,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=fa_version,
            )
            return out
        else:
            # 不捕获，但仍需更新计数器
            STREAMING_FRAME_ATTENTION_CAPTURE.current_layer_idx += 1

    # 检查是否需要捕获完整注意力权重（旧方式，可能 OOM）
    if ATTENTION_WEIGHT_CAPTURE.enabled and ATTENTION_WEIGHT_CAPTURE.should_capture():
        out, attn_data = attention_with_weights(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
            return_logits=ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 根据配置返回 logits 或 probs
        )
        # 存储注意力权重（移到 CPU 以节省 GPU 内存）
        ATTENTION_WEIGHT_CAPTURE.captured_weights.append({
            'layer_idx': ATTENTION_WEIGHT_CAPTURE.get_effective_layer_idx(),  # 使用模块化索引
            'attn_weights': attn_data.cpu(),
            'q_shape': q.shape,
            'k_shape': k.shape,
            'is_logits': ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 标记是 logits 还是 probs
        })
        ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1
        return out

    ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    return_logits=True,
):
    """
    计算注意力并返回注意力权重。
    这比 flash attention 慢，但允许我们捕获注意力权重用于可视化。

    Args:
        q: Query 张量，形状 [B, Lq, Nq, C]
        k: Key 张量，形状 [B, Lk, Nk, C]
        v: Value 张量，形状 [B, Lk, Nk, C]
        return_logits: 如果 True，返回 pre-softmax logits（用于 Figure 4）；
                      否则返回 post-softmax 概率

    Returns:
        out: 输出张量，形状 [B, Lq, Nq, C]
        attn_data: 注意力数据，形状 [B, Nq, Lq, Lk]
                  如果 return_logits=True，这是 pre-softmax 分数（可以是负值）
                  如果 return_logits=False，这是 post-softmax 概率 [0,1]
    """
    out_dtype = q.dtype

    # q: [B, Lq, N, C] -> [B, N, Lq, C]
    # k: [B, Lk, N, C] -> [B, N, Lk, C]
    # v: [B, Lk, N, C] -> [B, N, Lk, C]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # Support GQA/MQA: Q heads can be a multiple of K/V heads (Nq must be divisible by Nk).
    if q.shape[1] != k.shape[1]:
        n_q, n_k = q.shape[1], k.shape[1]
        if n_q % n_k != 0:
            raise ValueError(f"Nq must be divisible by Nk, got Nq={n_q}, Nk={n_k}")
        repeat_factor = n_q // n_k
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # 计算缩放因子
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # 计算注意力分数（logits）: [B, N, Lq, Lk]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    bsz, _n, lq, lk = attn_scores.shape

    # Apply key padding mask if provided (k_lens is [B]).
    q_valid = None
    if k_lens is not None:
        key_idx = torch.arange(lk, device=attn_scores.device).view(1, 1, 1, lk)
        key_valid = key_idx < k_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~key_valid, float('-inf'))

    # Track query validity to avoid NaNs when a padded query would be fully masked.
    if q_lens is not None:
        q_idx = torch.arange(lq, device=attn_scores.device).view(1, 1, lq, 1)
        q_valid = q_idx < q_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~q_valid, 0.0)

    # 对齐非方阵 Q/K：query i 对应 key i + (lk - lq)。
    # 对于 varlen（k_lens/q_lens）场景，flash-attn 使用每个样本的有效长度来计算 offset，
    # 否则 window/causal 的对齐会与快路径不一致。
    if (k_lens is not None) or (q_lens is not None):
        lk_eff = k_lens if k_lens is not None else torch.full((bsz,), lk, device=attn_scores.device, dtype=torch.long)
        lq_eff = q_lens if q_lens is not None else torch.full((bsz,), lq, device=attn_scores.device, dtype=torch.long)
        offset = (lk_eff - lq_eff).view(bsz, 1, 1)  # [B,1,1]
    else:
        offset = lk - lq  # scalar

    q_pos = torch.arange(lq, device=attn_scores.device).view(1, lq, 1)  # [1,Lq,1]
    k_pos = torch.arange(lk, device=attn_scores.device).view(1, 1, lk)  # [1,1,Lk]
    center = q_pos + offset  # scalar or [B,1,1] -> [B,Lq,1]

    # 如果需要 causal mask
    if causal:
        # Mask positions where key is "in the future" relative to the aligned center.
        causal_mask = k_pos > center  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))

    # Sliding window local attention (if enabled).
    # Semantics follow the same "offset" convention as the causal mask for non-square Q/K:
    # query position i is aligned to key position i + (lk - lq) (varlen uses per-sample offset).
    if window_size != (-1, -1):
        left, right = window_size
        if left < 0:
            left = lk
        if right < 0:
            right = lk
        lower = center - left
        upper = center + right
        window_mask = (k_pos < lower) | (k_pos > upper)  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(window_mask.unsqueeze(1), float('-inf'))

    # 计算注意力权重（概率）
    # 当 key padding mask + window mask（或 causal mask）导致某些 query 行被完全屏蔽时，
    # softmax(-inf, -inf, ...) 会产生 NaN；flash-attn 在这种情况下会输出 0。
    row_has_any_valid = torch.isfinite(attn_scores).any(dim=-1, keepdim=True)
    attn_scores_for_softmax = attn_scores.masked_fill(~row_has_any_valid, 0.0)
    attn_weights = torch.softmax(attn_scores_for_softmax, dim=-1)
    attn_weights = attn_weights.masked_fill(~row_has_any_valid, 0.0)

    # 应用 dropout
    if dropout_p > 0.:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # 计算输出: [B, N, Lq, C]
    out = torch.matmul(attn_weights, v)

    if q_valid is not None:
        out = out.masked_fill(~q_valid, 0.0)
        attn_weights = attn_weights.masked_fill(~q_valid, 0.0)

    # 转置回来: [B, N, Lq, C] -> [B, Lq, N, C]
    out = out.transpose(1, 2).contiguous().to(out_dtype)

    # 根据配置返回 logits 或 probs
    if return_logits:
        return out, attn_scores  # 返回 pre-softmax logits
    else:
        return out, attn_weights  # 返回 post-softmax probs
