# Attention Score 提取分析

本文档分析了 Wan2.1 模型中 Attention 模块的位置、输入输出维度，并提供了用于提取 Pre-softmax Attention Score (QK^T) 的 Monkey Patch Hook 代码示例。

## 1. Attention 模块定位

**核心文件与调用位置：**

| 文件 | 类/函数 | 行号 | 说明 |
|------|---------|------|------|
| `wan/modules/attention.py` | `flash_attention()` | 32-136 | 使用 `flash_attn.flash_attn_varlen_func` |
| `wan/modules/attention.py` | `attention()` | 139-185 | 封装函数，无 FlashAttn 时回退到 `F.scaled_dot_product_attention` (line 181) |
| `wan/modules/causal_model.py` | `CausalWanSelfAttention.forward()` | 86-240 | **推理时使用**，调用 `attention()` 在 line 229 |
| `wan/modules/model.py` | `WanSelfAttention.forward()` | 127-156 | 非因果模型，调用 `flash_attention()` 在 line 146 |

## 2. Q, K, V 维度形状

根据 `wan/modules/attention.py:47-50` 的文档注释：

```python
"""
q:              [B, Lq, Nq, C1].   # [batch, query_seq_len, num_heads, head_dim]
k:              [B, Lk, Nk, C1].   # [batch, key_seq_len, num_heads, head_dim]
v:              [B, Lk, Nk, C2].   # [batch, key_seq_len, num_heads, head_dim]
"""
```

**维度格式是 `[batch, seq_len, heads, dim]`**，不是 `[batch, heads, seq_len, dim]`。

注意：在 `attention()` 函数中，当使用 `F.scaled_dot_product_attention` 时会先转置成 `[B, heads, seq_len, dim]`（line 177-179）。

## 3. Monkey Patch Hook 代码示例

### 方案一：Hook attention() 函数 (推荐)

适用于推理时提取所有 attention score

```python
import torch
import wan.modules.attention as attention_module

# 全局变量存储 attention scores
ATTENTION_SCORES = []

def attention_with_score_extraction(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1),
    deterministic=False, dtype=torch.bfloat16, fa_version=None,
):
    """
    替换 attention 函数，手动计算 QK^T 并保存
    输入维度: q, k, v 都是 [B, L, N, D] (batch, seq_len, num_heads, head_dim)
    """
    global ATTENTION_SCORES

    # 转换为 [B, N, L, D] 格式用于 bmm
    q_t = q.transpose(1, 2).to(dtype)  # [B, N, Lq, D]
    k_t = k.transpose(1, 2).to(dtype)  # [B, N, Lk, D]
    v_t = v.transpose(1, 2).to(dtype)  # [B, N, Lk, D]

    B, N, Lq, D = q_t.shape
    Lk = k_t.shape[2]

    # 计算 softmax_scale (如果未提供)
    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    # 计算 pre-softmax attention scores: QK^T
    # [B, N, Lq, D] @ [B, N, D, Lk] -> [B, N, Lq, Lk]
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale

    # 保存 pre-softmax scores (这就是 Attention Sink 实验需要的)
    ATTENTION_SCORES.append({
        'scores': attn_scores.detach().cpu(),  # [B, N, Lq, Lk]
        'shape': {'B': B, 'N': N, 'Lq': Lq, 'Lk': Lk, 'D': D}
    })

    # 应用 causal mask (如果需要)
    if causal:
        causal_mask = torch.triu(
            torch.ones(Lq, Lk, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)

    if dropout_p > 0:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p)

    # 计算输出: [B, N, Lq, Lk] @ [B, N, Lk, D] -> [B, N, Lq, D]
    out = torch.matmul(attn_probs, v_t)

    # 转回 [B, L, N, D] 格式
    out = out.transpose(1, 2).contiguous()

    return out


def install_attention_hook():
    """安装 Hook"""
    attention_module._original_attention = attention_module.attention
    attention_module.attention = attention_with_score_extraction
    print("✓ Attention hook installed")


def uninstall_attention_hook():
    """卸载 Hook"""
    if hasattr(attention_module, '_original_attention'):
        attention_module.attention = attention_module._original_attention
        del attention_module._original_attention
        print("✓ Attention hook uninstalled")


def clear_attention_scores():
    """清空保存的 scores"""
    global ATTENTION_SCORES
    ATTENTION_SCORES = []


def get_attention_scores():
    """获取保存的 scores"""
    return ATTENTION_SCORES
```

### 方案二：Hook CausalWanSelfAttention 类的 forward 方法

可以更精细地控制每一层

```python
from wan.modules.causal_model import CausalWanSelfAttention

LAYER_ATTENTION_SCORES = {}

def make_hooked_forward(original_forward, layer_idx):
    """为特定层创建带 hook 的 forward"""

    def hooked_forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                       kv_cache=None, current_start=0, cache_start=None):
        global LAYER_ATTENTION_SCORES

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # 计算 Q, K, V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # 手动计算并保存 attention scores
        q_t = q.transpose(1, 2)  # [B, N, L, D]
        k_t = k.transpose(1, 2)

        softmax_scale = 1.0 / (d ** 0.5)
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale

        if layer_idx not in LAYER_ATTENTION_SCORES:
            LAYER_ATTENTION_SCORES[layer_idx] = []
        LAYER_ATTENTION_SCORES[layer_idx].append(attn_scores.detach().cpu())

        # 调用原始 forward (使用 FlashAttention)
        return original_forward(x, seq_lens, grid_sizes, freqs, block_mask,
                               kv_cache, current_start, cache_start)

    return hooked_forward


def install_layer_hooks(model, layer_indices=None):
    """
    为模型的特定层安装 hooks

    Args:
        model: CausalWanModel 实例
        layer_indices: 要 hook 的层索引列表，None 表示所有层
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.blocks)))

    for idx in layer_indices:
        block = model.blocks[idx]
        self_attn = block.self_attn

        # 保存原始 forward
        if not hasattr(self_attn, '_original_forward'):
            self_attn._original_forward = self_attn.forward

        # 创建绑定方法
        import types
        hooked = make_hooked_forward(self_attn._original_forward, idx)
        self_attn.forward = types.MethodType(hooked, self_attn)

    print(f"✓ Installed hooks on layers: {layer_indices}")
```

## 4. 使用示例

```python
# === 使用方案一 ===
install_attention_hook()

# 运行推理
# ... your inference code ...

# 获取 attention scores
scores = get_attention_scores()
for i, s in enumerate(scores):
    print(f"Layer {i}: {s['shape']}")
    # s['scores'] 就是 pre-softmax QK^T, shape [B, heads, Lq, Lk]

# 可视化 Attention Sink
import matplotlib.pyplot as plt
import numpy as np

# 取第一层、第一个 batch、第一个 head
attn = scores[0]['scores'][0, 0].numpy()  # [Lq, Lk]
plt.figure(figsize=(10, 8))
plt.imshow(attn, aspect='auto', cmap='viridis')
plt.colorbar(label='Pre-softmax Score (QK^T)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Score Heatmap (Layer 0, Head 0)')
plt.savefig('attention_sink.png')

uninstall_attention_hook()
```

## 5. 关键点总结

| 项目 | 详情 |
|------|------|
| **最佳 Hook 位置** | `wan/modules/attention.py:139` 的 `attention()` 函数 |
| **推理时调用点** | `wan/modules/causal_model.py:229` |
| **Q/K/V 维度** | `[B, seq_len, num_heads, head_dim]` |
| **Wan 1.3B 参数** | `num_heads=12`, `head_dim=128` (dim=1536) |
| **每帧 token 数** | `frame_seqlen = 60 * 104 / 4 = 1560` |

## 6. 注意事项

- 方案一直接替换全局的 `attention()` 函数，但会**禁用 FlashAttention**，显存占用增大
- 如果只需要可视化少量样本，建议使用小 batch size 和少量帧数
- Pre-softmax score 就是论文中用于证明 Attention Sink 的 `QK^T / sqrt(d)`
