# 测试模块文档

本文档描述 `tests/` 目录下的测试结构和逻辑。

## 目录结构

```
tests/
├── __init__.py                    # 包初始化
├── conftest.py                    # pytest 配置和 hooks
├── test_extraction_logic.py       # 注意力提取逻辑测试 (主要)
├── test_attention_equivalence.py  # 注意力等价性测试
├── test_attention_extraction.py   # 注意力提取集成测试
└── test_flash.py                  # Flash Attention 测试
```

## 运行测试

```bash
# 运行所有单元测试（跳过 GPU 测试）
python -m pytest tests -v

# 运行特定测试文件
python -m pytest tests/test_extraction_logic.py -v

# 运行 GPU 集成测试（需要 CUDA）
python -m pytest tests --run-slow -v

# 只运行特定测试类
python -m pytest tests/test_extraction_logic.py::TestBlockStructure -v

# 显示详细输出
python -m pytest tests -v -s
```

## conftest.py 配置

### 自定义命令行选项

```python
--run-slow  # 运行标记为 slow 的 GPU 测试
```

### 测试标记 (Markers)

| 标记 | 说明 |
|-----|------|
| `@pytest.mark.slow` | 慢速测试，需要 `--run-slow` 才运行 |
| `@pytest.mark.gpu` | 需要 GPU 的测试 |

### 进度提示 Hooks

- **pytest_runtest_setup**: GPU 测试开始时显示提示
- **pytest_runtest_teardown**: 显示耗时 >5s 的测试时间
- **pytest_report_teststatus**: 自定义状态符号 (✓/✗/○)

## test_extraction_logic.py

主要测试文件，包含 6 个测试类。

### TestBlockStructure

测试 block 结构计算逻辑。

```python
class TestBlockStructure:
    def test_block_sizes_with_independent_first_frame(self):
        """测试 independent_first_frame=True 时的 block 结构

        输入: num_frames=21, num_frame_per_block=3, independent_first_frame=True
        期望: block_sizes = [1, 3, 3, 3, 3, 3, 3], sum=19
        """

    def test_block_sizes_without_independent_first_frame(self):
        """测试 independent_first_frame=False 时的 block 结构

        输入: num_frames=21, num_frame_per_block=3
        期望: block_sizes = [3, 3, 3, 3, 3, 3, 3], sum=21
        """

    def test_cumulative_k_frames(self):
        """测试每个 block 对应的累积 K 帧数

        block_sizes = [1, 3, 3, 3, 3, 3, 3]
        期望累积: [1, 4, 7, 10, 13, 16, 19]
        """
```

### TestIndexMapping

测试索引映射逻辑。

```python
class TestIndexMapping:
    def test_layer_to_self_attn_index(self):
        """测试 layer index 到 self-attention 调用索引的映射

        规则: layer N → self-attn 调用索引 2*N
        原因: 每个 block 有 self-attn (偶数) 和 cross-attn (奇数)

        示例:
          layer 0  → call index 0
          layer 4  → call index 8
          layer 15 → call index 30
          layer 29 → call index 58
        """

    def test_call_index_to_block_index(self):
        """测试调用索引回转到 block 索引

        规则: call_idx // 2 = block_idx
        """
```

### TestDataFormat

测试输出数据格式。

```python
class TestDataFormat:
    def test_output_data_structure(self):
        """测试输出数据结构

        必要字段:
          - layer_index: int
          - full_frame_attention: [num_heads, num_frames, num_frames]
          - last_block_frame_attention: [num_heads, num_frames]
          - num_frames: int
          - num_heads: int
          - block_sizes: list
          - last_block_query_frames: list
        """

    def test_attention_matrix_is_causal(self):
        """测试注意力矩阵是因果的（下三角）

        验证: 上三角 (k > q) 应该为零
        """
```

### TestAttentionCaptureMechanism

测试 `ATTENTION_WEIGHT_CAPTURE` 全局捕获机制。

```python
class TestAttentionCaptureMechanism:
    def test_enable_disable(self):
        """测试启用和禁用

        ATTENTION_WEIGHT_CAPTURE.enable(layer_indices=[0, 4], num_layers=30)
        ATTENTION_WEIGHT_CAPTURE.disable()
        """

    def test_should_capture_logic(self):
        """测试 should_capture 逻辑

        使用模块化索引: current_layer_idx % num_layers

        示例 (layer_indices=[0, 8], num_layers=60):
          idx=0   → 0 % 60 = 0  ✓ 捕获
          idx=8   → 8 % 60 = 8  ✓ 捕获
          idx=60  → 60 % 60 = 0 ✓ 捕获
          idx=1   → 1 % 60 = 1  ✗ 不捕获
        """

    def test_effective_layer_idx(self):
        """测试 effective layer index 计算

        effective = current_layer_idx % num_layers
        """
```

### TestFrameAttentionComputation

测试帧级注意力计算。

```python
class TestFrameAttentionComputation:
    def test_token_to_frame_aggregation(self):
        """测试 token 级注意力到 frame 级的聚合

        frame_seq_length = 1560 tokens/帧

        聚合方法:
          1. 对所有 query tokens 取平均
          2. 对每个 key frame 的 tokens 取平均
        """

    def test_full_matrix_assembly(self):
        """测试完整矩阵组装逻辑

        Block-based Causality (非严格 frame-level):
          - Block 内所有 Q frames 可以看到该 block 结束为止的所有 K frames
          - 例: Block 1 的 Q frames 1-3 都可以看到 K frames 0-3

        验证:
          - K 范围内应该有值
          - K 范围外应该为零
        """
```

### TestIntegrationWithGPU

GPU 集成测试（需要 `--run-slow`）。

```python
@pytest.mark.slow
@pytest.mark.gpu
class TestIntegrationWithGPU:
    def test_extraction_produces_valid_output(self):
        """测试提取脚本产生有效输出

        步骤:
          1. 加载配置和 pipeline
          2. 启用 ATTENTION_WEIGHT_CAPTURE
          3. 运行推理
          4. 验证捕获的数据格式

        验证:
          - 捕获了 attention 数据
          - 每个 attention 有 attn_weights, k_shape, q_shape
          - attn_weights 是 4D tensor [B, N, Lq, Lk]
        """

    def test_full_matrix_shape(self):
        """测试完整矩阵形状正确

        步骤:
          1. 捕获 attention
          2. 按 K 长度排序
          3. 逐 block 组装完整矩阵

        验证:
          - 形状为 [num_heads, num_frames, num_frames]
          - 矩阵不全为零
        """
```

## 测试逻辑详解

### Block 结构计算

```python
num_frames = 21
num_frame_per_block = 3
independent_first_frame = True

if independent_first_frame:
    # 第一帧独立，剩余帧按 block 大小分组
    num_blocks = (num_frames - 1) // num_frame_per_block + 1
    block_sizes = [1] + [num_frame_per_block] * ((num_frames - 1) // num_frame_per_block)
    # = [1, 3, 3, 3, 3, 3, 3], sum = 19
else:
    num_blocks = num_frames // num_frame_per_block
    block_sizes = [num_frame_per_block] * num_blocks
    # = [3, 3, 3, 3, 3, 3, 3], sum = 21
```

### Self-Attention 索引映射

每个 Transformer block 有 2 次 attention 调用：

```
调用 0: self-attention (layer 0)
调用 1: cross-attention (layer 0)
调用 2: self-attention (layer 1)
调用 3: cross-attention (layer 1)
...
调用 2N: self-attention (layer N)
调用 2N+1: cross-attention (layer N)
```

因此：
```python
layer_index = 3
self_attn_call_index = 2 * layer_index  # = 6
```

### Block-based Causality

与传统 causal attention 不同：

| 传统 Causal | Block-based Causal |
|------------|-------------------|
| frame q 只能看 k ≤ q | block 内所有 q 可以看到 block 结束为止的 k |
| 严格下三角 | 阶梯状下三角 |

```
Block 0: Q=[0],       K=[0]           (1帧)
Block 1: Q=[1,2,3],   K=[0,1,2,3]     (4帧)
Block 2: Q=[4,5,6],   K=[0,1,2,3,4,5,6] (7帧)
...
```

### Token-to-Frame 聚合

```python
frame_seq_length = 1560  # tokens per frame

# 对于 Q frame qf 和 K frame kf
q_start = qf * frame_seq_length
q_end = (qf + 1) * frame_seq_length
k_start = kf * frame_seq_length
k_end = (kf + 1) * frame_seq_length

# 取所有 token pair 的平均
frame_attention[h, qf, kf] = attn_logits[h, q_start:q_end, k_start:k_end].mean()
```

## 进度提示输出示例

```
tests/test_extraction_logic.py::TestIntegrationWithGPU::test_extraction_produces_valid_output
============================================================
[GPU TEST] test_extraction_produces_valid_output
============================================================
⏳ 加载模型中... (首次运行 torch.compile 需要 5-10 分钟)

📁 加载配置文件...
🔧 初始化 pipeline...
✓ Pipeline 初始化完成
🚀 运行推理 (首次运行需要编译，请耐心等待)...
✓ 推理完成，捕获了 7 个 attention
✓ 集成测试通过: 捕获了 7 个 attention

⏱️  test_extraction_produces_valid_output 耗时: 312.5s
PASSED
```

## 相关文件

- `tests/conftest.py` - pytest 配置
- `tests/test_extraction_logic.py` - 主测试文件
- `wan/modules/attention.py` - ATTENTION_WEIGHT_CAPTURE 实现
- `experiments/run_extraction_each.py` - 被测试的提取脚本
- `docs/attention_extraction.md` - 提取逻辑文档
