# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
分布式流式帧级注意力提取模块。

数学逻辑与 attention_with_weights 严格一致：
1. 转置 q, k, v 到 [B, N, L, C] 格式
2. 应用 q_scale
3. 处理 GQA/MQA（repeat K/V heads）
4. 计算 softmax_scale
5. 计算 attn_scores = Q @ K^T * scale
6. 应用 key padding mask (k_lens)
7. 应用 query padding mask (q_lens)
8. 计算 offset 用于对齐非方阵 Q/K
9. 应用 causal mask
10. 应用 window mask

分块策略：
- 为避免 OOM，分块计算 Q_frame @ K_chunk^T
- 每块计算完成后立即聚合到 frame-level 并释放
"""

import torch
import torch.distributed as dist
import gc
from typing import Optional, Callable, List, Tuple

__all__ = [
    'DistributedStreamingFrameAttentionCapture',
    'DISTRIBUTED_STREAMING_CAPTURE',
]


class DistributedStreamingFrameAttentionCapture:
    """
    分布式流式帧级注意力捕获器。

    数学等价性保证：
    - 与 attention_with_weights 完全相同的 attention score 计算
    - 支持 causal mask, window mask, k_lens, q_lens
    - 分块计算以控制显存
    """

    def __init__(self):
        self.enabled = False
        self.layer_indices = None
        self.current_layer_idx = 0
        self.num_layers = 60  # 30 blocks × 2 (self + cross)

        # Frame-level 配置
        self.frame_seq_length = 1560
        self.num_heads = 12
        self.chunk_frames = 3

        # 分布式配置
        self.world_size = 1
        self.rank = 0
        self.total_frames = 0

        # 回调函数
        self.on_frame_attention = None

    def enable(
        self,
        layer_indices: Optional[List[int]] = None,
        num_layers: int = 60,
        frame_seq_length: int = 1560,
        num_heads: int = 12,
        chunk_frames: int = 3,
        total_frames: int = 120,
        on_frame_attention: Optional[Callable] = None,
    ):
        """启用分布式流式帧级注意力捕获。"""
        self.enabled = True
        self.layer_indices = layer_indices
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.chunk_frames = chunk_frames
        self.total_frames = total_frames
        self.on_frame_attention = on_frame_attention
        self.current_layer_idx = 0

        # 初始化分布式配置
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def disable(self):
        """禁用捕获。"""
        self.enabled = False
        self.current_layer_idx = 0
        self.on_frame_attention = None

    def should_capture(self) -> bool:
        """检查是否应该捕获当前层。"""
        if not self.enabled:
            return False
        if self.layer_indices is None:
            return True
        effective_idx = self.current_layer_idx % self.num_layers
        return effective_idx in self.layer_indices

    def get_effective_layer_idx(self) -> int:
        """获取当前的有效层索引。"""
        return self.current_layer_idx % self.num_layers

    def compute_frame_attention_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        softmax_scale: Optional[float] = None,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        分块计算 frame-level attention logits。

        数学逻辑与 attention_with_weights 严格一致。

        Args:
            q: Query 张量 [B, Lq, N, C]（原始格式，未转置）
            k: Key 张量 [B, Lk, N, C]（原始格式，未转置）
            softmax_scale: 缩放因子
            q_lens: Query 有效长度 [B]
            k_lens: Key 有效长度 [B]
            causal: 是否应用 causal mask
            window_size: 滑动窗口大小 (left, right)
            dtype: 计算精度

        Returns:
            frame_attn: [N, q_frames, k_frames] 的 frame-level attention logits（均值聚合）
        """
        # ========== Step 1: 转置到 [B, N, L, C] ==========
        # 与 attention_with_weights 第 546-548 行一致
        q_t = q.transpose(1, 2).to(dtype)  # [B, N, Lq, C]
        k_t = k.transpose(1, 2).to(dtype)  # [B, N, Lk, C]

        b, n_q, lq, c = q_t.shape
        _, n_k, lk, _ = k_t.shape

        # ========== Step 2: 处理 GQA/MQA ==========
        # 与 attention_with_weights 第 554-560 行一致
        if n_q != n_k:
            if n_q % n_k != 0:
                raise ValueError(f"Nq must be divisible by Nk, got Nq={n_q}, Nk={n_k}")
            repeat_factor = n_q // n_k
            k_t = k_t.repeat_interleave(repeat_factor, dim=1)
            n_k = n_q

        # ========== Step 3: 计算 softmax_scale ==========
        # 与 attention_with_weights 第 563-564 行一致
        if softmax_scale is None:
            softmax_scale = c ** -0.5

        # ========== Step 4: 计算 offset（用于 causal/window mask 对齐）==========
        # 与 attention_with_weights 第 587-596 行一致
        device = q_t.device
        bsz = b

        if (k_lens is not None) or (q_lens is not None):
            lk_eff = k_lens if k_lens is not None else torch.full((bsz,), lk, device=device, dtype=torch.long)
            lq_eff = q_lens if q_lens is not None else torch.full((bsz,), lq, device=device, dtype=torch.long)
            offset = (lk_eff - lq_eff).view(bsz, 1, 1)  # [B, 1, 1]
        else:
            offset = lk - lq  # scalar

        # ========== Step 5: 分块计算 attention scores ==========
        q_frames = lq // self.frame_seq_length
        k_frames = lk // self.frame_seq_length

        # 输出：frame-level attention [N, q_frames, k_frames]
        frame_attn = torch.zeros(n_q, q_frames, k_frames, dtype=torch.float32, device='cpu')

        # 分块处理 K frames
        for kf_start in range(0, k_frames, self.chunk_frames):
            kf_end = min(kf_start + self.chunk_frames, k_frames)
            k_token_start = kf_start * self.frame_seq_length
            k_token_end = kf_end * self.frame_seq_length

            # 取出 K chunk: [B, N, chunk_tokens, C]
            k_chunk = k_t[:, :, k_token_start:k_token_end, :]

            # 对每个 Q frame 计算 attention
            for qf in range(q_frames):
                q_token_start = qf * self.frame_seq_length
                q_token_end = (qf + 1) * self.frame_seq_length

                # 取出 Q frame: [B, N, frame_tokens, C]
                q_frame = q_t[:, :, q_token_start:q_token_end, :]

                # ========== 计算 attention scores ==========
                # 与 attention_with_weights 第 567 行一致
                # scores: [B, N, frame_tokens, chunk_tokens]
                scores = torch.matmul(q_frame.float(), k_chunk.float().transpose(-2, -1)) * softmax_scale

                local_lq = q_frame.shape[2]  # frame_tokens
                local_lk = k_chunk.shape[2]  # chunk_tokens

                # ========== 应用 key padding mask ==========
                # 与 attention_with_weights 第 573-576 行一致
                if k_lens is not None:
                    # 全局 key 索引
                    key_idx = torch.arange(k_token_start, k_token_end, device=device).view(1, 1, 1, local_lk)
                    key_valid = key_idx < k_lens.view(bsz, 1, 1, 1)
                    scores = scores.masked_fill(~key_valid, float('-inf'))

                # ========== 应用 query padding mask ==========
                # 与 attention_with_weights 第 579-582 行一致
                if q_lens is not None:
                    # 全局 query 索引
                    q_idx = torch.arange(q_token_start, q_token_end, device=device).view(1, 1, local_lq, 1)
                    q_valid = q_idx < q_lens.view(bsz, 1, 1, 1)
                    scores = scores.masked_fill(~q_valid, 0.0)

                # ========== 应用 causal mask ==========
                # 与 attention_with_weights 第 599-602 行一致
                if causal:
                    # 全局位置索引
                    q_pos = torch.arange(q_token_start, q_token_end, device=device).view(1, local_lq, 1)
                    k_pos = torch.arange(k_token_start, k_token_end, device=device).view(1, 1, local_lk)
                    center = q_pos + offset  # [B, Lq, 1] or [1, Lq, 1]

                    # Mask positions where key is "in the future"
                    causal_mask = k_pos > center  # [B, Lq, Lk] or [1, Lq, Lk]
                    scores = scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))

                # ========== 应用 window mask ==========
                # 与 attention_with_weights 第 607-616 行一致
                if window_size != (-1, -1):
                    left, right = window_size
                    if left < 0:
                        left = lk
                    if right < 0:
                        right = lk

                    q_pos = torch.arange(q_token_start, q_token_end, device=device).view(1, local_lq, 1)
                    k_pos = torch.arange(k_token_start, k_token_end, device=device).view(1, 1, local_lk)
                    center = q_pos + offset

                    lower = center - left
                    upper = center + right
                    window_mask = (k_pos < lower) | (k_pos > upper)
                    scores = scores.masked_fill(window_mask.unsqueeze(1), float('-inf'))

                # ========== 聚合到 frame-level ==========
                # 对每个 K frame 分别计算均值
                for kf_local in range(kf_end - kf_start):
                    kf = kf_start + kf_local
                    k_local_start = kf_local * self.frame_seq_length
                    k_local_end = (kf_local + 1) * self.frame_seq_length

                    # 取出这个 K frame 对应的 scores: [B, N, 1560, 1560]
                    frame_scores = scores[:, :, :, k_local_start:k_local_end]

                    # 计算均值：对 batch, q_tokens, k_tokens 维度取平均
                    # 结果：[N] -> 存入 frame_attn[:, qf, kf]
                    frame_attn[:, qf, kf] = frame_scores.mean(dim=(0, 2, 3)).cpu()

                # 释放中间结果
                del scores

            # 释放 K chunk
            del k_chunk
            torch.cuda.empty_cache()

        return frame_attn

    def capture_and_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        flash_attn_fn: Callable,
        **kwargs,
    ) -> torch.Tensor:
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
            # 提取 kwargs 中的参数
            q_scale = kwargs.get('q_scale')
            softmax_scale = kwargs.get('softmax_scale')
            q_lens = kwargs.get('q_lens')
            k_lens = kwargs.get('k_lens')
            causal = kwargs.get('causal', False)
            window_size = kwargs.get('window_size', (-1, -1))
            dtype = kwargs.get('dtype', torch.bfloat16)

            # 应用 q_scale（与 attention_with_weights 第 550-551 行一致）
            q_for_attn = q
            if q_scale is not None:
                q_for_attn = q * q_scale

            # 计算 frame-level attention
            frame_attn = self.compute_frame_attention_chunked(
                q=q_for_attn,
                k=k,
                softmax_scale=softmax_scale,
                q_lens=q_lens,
                k_lens=k_lens,
                causal=causal,
                window_size=window_size,
                dtype=dtype,
            )

            # 调用回调
            if self.on_frame_attention is not None:
                q_frames = q.shape[1] // self.frame_seq_length
                k_frames = k.shape[1] // self.frame_seq_length
                self.on_frame_attention(
                    layer_idx=self.get_effective_layer_idx(),
                    frame_attn=frame_attn,
                    q_frames=q_frames,
                    k_frames=k_frames,
                    rank=self.rank,
                )

            # 清理
            del frame_attn
            if q_scale is not None:
                del q_for_attn
            gc.collect()
            torch.cuda.empty_cache()

        self.current_layer_idx += 1
        return out


# 全局实例
DISTRIBUTED_STREAMING_CAPTURE = DistributedStreamingFrameAttentionCapture()


class DistributedFrameAttentionAggregator:
    """
    分布式帧级注意力聚合器。

    每个 rank 收集注意力数据，支持分布式场景下的结果合并。
    """

    def __init__(
        self,
        num_frames: int,
        num_heads: int,
        block_sizes: List[int],
        rank: int = 0,
        world_size: int = 1,
    ):
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.block_sizes = block_sizes
        self.rank = rank
        self.world_size = world_size

        # 完整的 frame-level attention
        self.full_frame_attn = torch.zeros(
            num_heads, num_frames, num_frames,
            dtype=torch.float32
        )

        # 追踪处理进度
        self.last_k_frames = 0
        self.current_q_start = 0
        self.blocks_processed = 0

    def on_frame_attention(
        self,
        layer_idx: int,
        frame_attn: torch.Tensor,
        q_frames: int,
        k_frames: int,
        rank: int = 0,
    ):
        """回调函数：接收流式的 frame-level attention 数据。"""
        if k_frames <= self.last_k_frames:
            return

        if self.blocks_processed >= len(self.block_sizes):
            return

        block_size = self.block_sizes[self.blocks_processed]

        if rank == 0 or self.rank == 0:
            print(f"  [Rank {self.rank}] Block {self.blocks_processed}: "
                  f"Q frames {self.current_q_start}-{self.current_q_start + q_frames - 1}, "
                  f"K frames 0-{k_frames - 1}")

        # 填入完整矩阵
        q_end = self.current_q_start + q_frames
        self.full_frame_attn[:, self.current_q_start:q_end, :k_frames] = frame_attn

        # 更新状态
        self.last_k_frames = k_frames
        self.current_q_start += block_size
        self.blocks_processed += 1

    def gather_results(self) -> torch.Tensor:
        """收集结果。"""
        if self.world_size == 1 or not dist.is_initialized():
            return self.full_frame_attn

        dist.barrier()
        return self.full_frame_attn


def estimate_distributed_memory(
    num_frames: int,
    world_size: int = 4,
    per_frame_mb: float = 450,
    base_gb: float = 3.0,
    save_overhead_gb: float = 6.0,
) -> dict:
    """
    估算分布式提取的显存需求。

    注意：由于 causal attention 需要完整 KV cache，
    分布式主要用于并行提取多层或多 prompt，而非分割单次推理。
    """
    runtime_gb = base_gb + num_frames * per_frame_mb / 1024
    peak_gb = runtime_gb + save_overhead_gb

    # 单卡最大帧数（假设 46GB L40）
    max_frames_single = int((46 * 0.9 - save_overhead_gb - base_gb) / (per_frame_mb / 1024))

    return {
        'num_frames': num_frames,
        'world_size': world_size,
        'runtime_gb': runtime_gb,
        'peak_gb': peak_gb,
        'max_frames_single_l40': max_frames_single,
        'can_run_single': peak_gb < 46 * 0.95,
    }
