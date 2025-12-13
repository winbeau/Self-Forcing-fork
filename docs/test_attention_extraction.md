# test_attention_extraction.py 详解

## 概述

本测试文件验证 `ATTENTION_WEIGHT_CAPTURE` 机制是否正常工作，确保注意力提取逻辑与模型实际推理一致。

**测试目标**：
1. 模块化索引正确工作
2. 能捕获多个 block/timestep 的 attention
3. 最后一个 block 的 K 包含所有历史帧

## 文件位置

```
experiments/test_attention_extraction.py
```

## 运行方式

```bash
# 运行所有测试
PYTHONPATH=. pytest experiments/test_attention_extraction.py -v -s

# 运行单个测试
PYTHONPATH=. pytest experiments/test_attention_extraction.py::TestAttentionCapture::test_modular_index -v -s

# 跳过需要 GPU 的测试
PYTHONPATH=. pytest experiments/test_attention_extraction.py -v -s -k "not cuda"
```

---

## 测试类结构

```python
class TestAttentionCapture:
    """测试 AttentionWeightCapture 类"""

    def test_modular_index(self):
        """测试模块化索引正确工作（纯逻辑测试，不需要 GPU）"""

    def test_capture_shape_grows_with_kv_cache(self):
        """测试 KV cache 模式下，K 的长度随着 block 增加而增长（需要 GPU）"""

    def test_last_block_has_full_history(self):
        """测试最后一个 block 的 attention 包含完整历史（需要 GPU）"""
```

---

## 测试 1: test_modular_index

### 目的

验证 `AttentionWeightCapture` 的模块化索引逻辑，确保在多个 denoising step 中能正确识别目标层。

### 背景问题

Wan 模型有 30 层。推理时每个 block 有多个 denoising step，`current_layer_idx` 会持续递增：

```
Step 0: layer 0, 1, 2, ..., 29    (current_layer_idx: 0-29)
Step 1: layer 0, 1, 2, ..., 29    (current_layer_idx: 30-59)
Step 2: layer 0, 1, 2, ..., 29    (current_layer_idx: 60-89)
...
```

如果不用模块化索引，`layer_indices=[0, 4]` 只会在第一个 step 匹配，后续 step 永远不会匹配。

### 关键代码

```python
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
```

### 逻辑详解

```python
# 启用捕获，指定层 [0, 4] 和总层数 30
capture.enable(layer_indices=[0, 4], num_layers=30)

# 模拟 current_layer_idx 的变化
# Step 0:
#   layer 0: current_layer_idx=0,  effective=0%30=0  ✓ 匹配
#   layer 4: current_layer_idx=4,  effective=4%30=4  ✓ 匹配
# Step 1:
#   layer 0: current_layer_idx=30, effective=30%30=0 ✓ 匹配
#   layer 4: current_layer_idx=34, effective=34%30=4 ✓ 匹配
# Step 2:
#   layer 0: current_layer_idx=60, effective=60%30=0 ✓ 匹配
#   layer 4: current_layer_idx=64, effective=64%30=4 ✓ 匹配

# 总共 3 步 × 2 层 = 6 次捕获
```

### 验证的核心逻辑（在 attention.py 中）

```python
def should_capture(self):
    """检查是否应该捕获当前层（使用模块化索引）"""
    if not self.enabled:
        return False
    if self.layer_indices is None:
        return True
    # 关键：使用模块化索引
    effective_layer_idx = self.current_layer_idx % self.num_layers
    return effective_layer_idx in self.layer_indices

def get_effective_layer_idx(self):
    """获取当前的有效层索引（模块化后）"""
    return self.current_layer_idx % self.num_layers
```

### 预期输出

```
✓ 模块化索引测试通过: 捕获了 6 个 attention
  捕获详情: [(0, 0, 0), (0, 4, 4), (1, 0, 0), (1, 4, 4), (2, 0, 0), (2, 4, 4)]
```

---

## 测试 2: test_capture_shape_grows_with_kv_cache

### 目的

验证在 KV cache 模式下，随着 block 推进，K 的长度会增长。

### 背景

Autoregressive 生成中，KV cache 会累积：

```
Block 1: 生成 frame 0-2,  K 缓存 frame 0-2   → K_len = 3 × 1560
Block 2: 生成 frame 3-5,  K 缓存 frame 0-5   → K_len = 6 × 1560
Block 3: 生成 frame 6-8,  K 缓存 frame 0-8   → K_len = 9 × 1560
...
Block 7: 生成 frame 18-20, K 缓存 frame 0-20 → K_len = 21 × 1560
```

### 关键代码

```python
def test_capture_shape_grows_with_kv_cache(self):
    """测试 KV cache 模式下，K 的长度随着 block 增加而增长"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # ... 模型初始化 ...

    num_frames = 21
    num_frame_per_block = config.get('num_frame_per_block', 3)

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

    # 分析 K 的长度变化
    k_lengths = [c['k_shape'][1] for c in captured]
    print(f"K 长度序列（前12个）: {k_lengths[:12]}")

    # K 长度应该随着 block 增加而增长
    frame_seq_length = 60 * 104 // 4  # 1560

    # 检查最大 K 长度
    max_k_len = max(k_lengths)
    min_k_len = (num_frames - num_frame_per_block) * frame_seq_length

    assert max_k_len >= min_k_len, \
        f"Max K length {max_k_len} is less than expected minimum {min_k_len}"
```

### 验证逻辑

```python
# 计算 frame_seq_length
frame_seq_length = 60 * 104 // 4  # = 1560 tokens/帧
# 解释：latent 60×104，patch 2×2，所以 (60/2) × (104/2) = 1560

# 最后一个 block（frame 18-20）的 K 应该至少包含前 18 帧
# 实际上会包含 21 帧（包括当前 block）
min_k_len = (21 - 3) * 1560  # = 28080
# 但实际 max_k_len = 21 * 1560 = 32760
```

### 预期输出

```
捕获到 28 个 attention 矩阵
K 长度序列（前12个）: [4680, 4680, 4680, 4680, 9360, 9360, 9360, 9360, 14040, 14040, 14040, 14040]
最大 K 长度: 32760
期望的 K 长度（包含所有帧）: 32760
✓ KV cache 测试通过: K 长度最大为 21 帧
```

**K 长度序列解读**：
- 前 4 个 `4680`：Block 1 的 4 个 denoising step，K = 3 帧
- 接下来 4 个 `9360`：Block 2，K = 6 帧
- 以此类推...

---

## 测试 3: test_last_block_has_full_history

### 目的

验证最后一个 block 的 attention 确实包含完整历史，且 Q/K 帧数符合预期。

### 这是最重要的测试

Figure 4 需要分析「最后 3 帧（Q）对所有历史帧（K）的注意力」，这个测试验证我们能正确捕获这些数据。

### 关键代码

```python
def test_last_block_has_full_history(self):
    """测试最后一个 block 的 attention 包含完整历史"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # ... 模型初始化 ...

    num_frames = 21
    num_frame_per_block = 3
    frame_seq_length = 1560

    # 捕获 layer 0 和 4
    ATTENTION_WEIGHT_CAPTURE.enable(
        layer_indices=[0, 4],
        capture_logits=True,
        num_layers=num_layers
    )

    try:
        pipeline.inference(...)
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
        expected_q = num_frame_per_block * frame_seq_length  # 3 × 1560 = 4680
        assert q_len == expected_q

        # K 应该至少是 18 帧（所有历史帧）
        min_k = (num_frames - num_frame_per_block) * frame_seq_length  # 18 × 1560
        assert k_len >= min_k

        num_q_frames = q_len // frame_seq_length
        num_k_frames = k_len // frame_seq_length
        print(f"Layer {layer_idx}: Q={num_q_frames}帧, K={num_k_frames}帧")
```

### 验证的数学关系

```
总帧数: 21
每 block 帧数: 3
frame_seq_length: 1560

最后一个 block:
  Q: frames 18, 19, 20 → Q_len = 3 × 1560 = 4680
  K: frames 0-20       → K_len = 21 × 1560 = 32760

验证条件:
  Q_len == 4680        ✓
  K_len >= 18 × 1560   ✓ (实际 K_len = 32760 > 28080)
```

### 预期输出

```
Layer 0: Q=3帧, K=21帧
Layer 4: Q=3帧, K=21帧
✓ 最后一个 block 包含完整历史测试通过
```

---

## 测试失败排查

### 问题 1: "Expected 6 captures, got 2"

**原因**：`should_capture()` 没有使用模块化索引

**修复**：
```python
# 错误写法
return self.current_layer_idx in self.layer_indices

# 正确写法
effective_layer_idx = self.current_layer_idx % self.num_layers
return effective_layer_idx in self.layer_indices
```

### 问题 2: "K length 4680 < expected minimum 28080"

**原因**：只捕获了第一个 block，没有捕获后续 block

**排查**：检查 `k_lengths` 序列是否单调递增

### 问题 3: "CUDA not available"

**原因**：测试环境没有 GPU

**解决**：这些测试会自动跳过，用 `pytest.skip()`

---

## 测试覆盖的关键逻辑

| 测试 | 验证内容 |
|------|----------|
| `test_modular_index` | `current_layer_idx % num_layers` 逻辑 |
| `test_capture_shape_grows_with_kv_cache` | KV cache 累积机制 |
| `test_last_block_has_full_history` | 最后 block 的 Q/K 维度 |

---

## 与 attention.py 的关系

测试验证的核心代码在 `wan/modules/attention.py`：

```python
def attention(...):
    # 检查是否需要捕获
    if ATTENTION_WEIGHT_CAPTURE.enabled and ATTENTION_WEIGHT_CAPTURE.should_capture():
        out, attn_data = attention_with_weights(...)

        # 存储（使用模块化索引）
        ATTENTION_WEIGHT_CAPTURE.captured_weights.append({
            'layer_idx': ATTENTION_WEIGHT_CAPTURE.get_effective_layer_idx(),
            'attn_weights': attn_data.cpu(),
            'q_shape': q.shape,
            'k_shape': k.shape,
            'is_logits': ATTENTION_WEIGHT_CAPTURE.capture_logits,
        })
        ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1
        return out

    # 正常路径（不捕获时也要递增计数器）
    ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1
    # ... flash attention ...
```

测试确保这段逻辑在各种情况下都能正确工作。
