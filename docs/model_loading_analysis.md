# Self-Forcing 模型加载逻辑分析

## 问题背景

分析以下命令加载的是哪个模型：

```bash
python experiments/run_extraction_figure4.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_path cache/attention_cache_figure4.pt \
    --layer_indices 0 4 \
    --no_checkpoint
```

---

## 完整的文件调用逻辑

### 第1步：入口点
**`experiments/run_extraction_figure4.py:49`**
```python
pipeline = CausalInferencePipeline(args=config, device=device)
```

### 第2步：CausalInferencePipeline 初始化
**`pipeline/causal_inference.py:20-21`**
```python
self.generator = WanDiffusionWrapper(
    **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
```
- 传入 `is_causal=True`

### 第3步：WanDiffusionWrapper 初始化
**`utils/wan_wrapper.py:126-128`**
```python
if is_causal:
    self.model = CausalWanModel.from_pretrained(
        f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
```
- `model_name` 默认值为 `"Wan2.1-T2V-1.3B"`（见 `wan_wrapper.py:118`）
- 所以路径是 `wan_models/Wan2.1-T2V-1.3B/`

### 第4步：CausalWanModel.from_pretrained
**`wan/modules/causal_model.py:369`** - 类定义
```python
class CausalWanModel(ModelMixin, ConfigMixin):
```

`from_pretrained` 方法来自 **diffusers 库的 `ModelMixin`**，它会：
1. 加载 `wan_models/Wan2.1-T2V-1.3B/config.json` 获取模型配置
2. 加载 `wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` 获取权重

### 第5步：--no_checkpoint 分支
**`experiments/run_extraction_figure4.py:52-53`**
```python
if args.no_checkpoint:
    print("使用原始 Wan2.1 基础模型（不加载任何 checkpoint）")
```
- 此时**跳过**加载 `checkpoints/self_forcing_dmd.pt`
- 模型保持原始 Wan2.1 的权重不变

---

## 加载的具体文件

| 组件 | 文件路径 |
|------|---------|
| Diffusion 模型配置 | `wan_models/Wan2.1-T2V-1.3B/config.json` |
| Diffusion 模型权重 | `wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` (5.3GB) |
| T5 文本编码器 | `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` (10.6GB) |
| VAE | `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (484MB) |
| Tokenizer | `wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/` |

---

## 总结

使用 `--no_checkpoint` 时，加载的是**未经 Self-Forcing 训练的原始 Wan2.1-T2V-1.3B 预训练模型**。如果不加 `--no_checkpoint`，则会额外加载 `checkpoints/self_forcing_dmd.pt` 覆盖 generator 的权重（`run_extraction_figure4.py:56-61`）。

---

## 如何使用 Self-Forcing 参数

去掉 `--no_checkpoint` 参数即可：

```bash
python experiments/run_extraction_figure4.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_path cache/attention_cache_self_forcing.pt \
    --layer_indices 0 4 \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --use_ema
```

### 对应的加载逻辑

**`experiments/run_extraction_figure4.py:54-61`**
```python
elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
    print(f"Loading checkpoint from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    key = 'generator_ema' if args.use_ema else 'generator'
    if key in state_dict:
        pipeline.generator.load_state_dict(state_dict[key])
    else:
        pipeline.generator.load_state_dict(state_dict['generator'])
```

### 参数说明

| 参数 | 作用 |
|------|------|
| `--checkpoint_path checkpoints/self_forcing_dmd.pt` | 指定 Self-Forcing 训练后的 checkpoint |
| `--use_ema` | 使用 EMA 权重 (`generator_ema`)，通常效果更好 |
| 不加 `--no_checkpoint` | 允许加载 checkpoint |

### 加载流程

1. 先加载原始 Wan2.1 基础模型 (`diffusion_pytorch_model.safetensors`)
2. 再用 `checkpoints/self_forcing_dmd.pt` 中的 `generator_ema` 权重**覆盖** generator

### 其他可用的 checkpoint

```bash
# 使用 SID checkpoint
python experiments/run_extraction_figure4.py \
    --config_path configs/self_forcing_sid.yaml \
    --output_path cache/attention_cache_self_forcing_sid.pt \
    --layer_indices 0 4 \
    --checkpoint_path checkpoints/self_forcing_sid.pt \
    --use_ema
```

---

## 表格1：命令行参数解析逻辑

| 顺序 | 参数名 | 类型 | 默认值 | 作用 |
|:---:|--------|------|--------|------|
| 1 | `--config_path` | str | `configs/self_forcing_dmd.yaml` | 模型配置文件路径 |
| 2 | `--checkpoint_path` | str | `checkpoints/self_forcing_dmd.pt` | Self-Forcing checkpoint 路径 |
| 3 | `--output_path` | str | `cache/attention_cache_figure4.pt` | 输出的 attention 缓存路径 |
| 4 | `--prompt` | str | `"A majestic eagle..."` | 生成视频的文本提示 |
| 5 | `--num_frames` | int | `21` | 生成的帧数 |
| 6 | `--layer_indices` | int[] | `[0, 4]` | 捕获 attention 的层索引 |
| 7 | `--no_checkpoint` | bool | `False` | 若为 True，跳过加载 checkpoint |
| 8 | `--use_ema` | bool | `True` | 若为 True，使用 EMA 权重 |
| 9 | `--seed` | int | `42` | 随机种子 |
| 10 | `--gpu_id` | int | `0` | 使用的 GPU ID |

**代码位置**: `run_extraction_figure4.py:224-238`

---

## 表格2：模型加载逻辑

| 模型组件 | 权重文件路径 | 加载逻辑详解 |
|----------|-------------|--------------|
| **配置文件** | `configs/self_forcing_dmd.yaml` + `configs/default_config.yaml` | **`run_extraction_figure4.py:42-44`**<br>1. 加载指定的 config_path<br>2. 加载 default_config.yaml<br>3. 使用 `OmegaConf.merge()` 合并，指定 config 覆盖默认值 |
| **Diffusion 模型 (CausalWanModel)** | `wan_models/Wan2.1-T2V-1.3B/config.json`<br>`wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` | **`wan_wrapper.py:126-128`**<br>1. `CausalWanModel.from_pretrained()` 从 diffusers 的 `ModelMixin` 继承<br>2. 读取 `config.json` 初始化模型结构（dim=1536, num_layers=30, num_heads=12）<br>3. 加载 `diffusion_pytorch_model.safetensors` (5.3GB) 作为基础权重 |
| **Self-Forcing Checkpoint** | `checkpoints/self_forcing_dmd.pt` | **`run_extraction_figure4.py:52-63`**<br>1. 若 `--no_checkpoint` 为 True → **跳过**，使用原始 Wan2.1 权重<br>2. 若 checkpoint 存在 → `torch.load()` 加载<br>3. 若 `--use_ema` 为 True → 取 `state_dict['generator_ema']`<br>4. 否则 → 取 `state_dict['generator']`<br>5. 调用 `pipeline.generator.load_state_dict()` **覆盖** Diffusion 模型权重 |
| **T5 文本编码器** | `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | **`wan_wrapper.py:18-27`**<br>1. `umt5_xxl()` 创建 T5 模型结构<br>2. `torch.load()` 加载 bf16 权重 (10.6GB)<br>3. 设置 `eval()` 和 `requires_grad_(False)` |
| **Tokenizer** | `wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/` | **`wan_wrapper.py:29-30`**<br>`HuggingfaceTokenizer` 从该目录加载 tokenizer 配置 |
| **VAE** | `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | **`wan_wrapper.py:68-71`**<br>1. `_video_vae()` 创建 VAE 结构<br>2. `pretrained_path` 参数直接加载权重 (484MB)<br>3. 设置 `eval()` 和 `requires_grad_(False)` |

---

## 表格3：Checkpoint 加载决策逻辑

| 条件判断 | 代码行号 | 结果 |
|----------|----------|------|
| `args.no_checkpoint == True` | `line 52-53` | 使用原始 Wan2.1 基础模型，不加载任何 checkpoint |
| `args.no_checkpoint == False` 且 `checkpoint 存在` | `line 54-61` | 加载 checkpoint 覆盖 generator 权重 |
| `args.no_checkpoint == False` 且 `checkpoint 不存在` | `line 62-63` | 打印警告，使用原始 Wan2.1 基础模型 |
| `args.use_ema == True` | `line 57-59` | 使用 `state_dict['generator_ema']` |
| `args.use_ema == False` | `line 57, 61` | 使用 `state_dict['generator']` |

---

## 模型加载顺序流程图

```
run_extraction_figure4.py:main()
│
├─[42-44] 加载配置 (OmegaConf)
│   └─ configs/self_forcing_dmd.yaml + configs/default_config.yaml
│
├─[49] CausalInferencePipeline(args, device)
│   │
│   ├─ pipeline/causal_inference.py:20-23
│   │   ├─ WanDiffusionWrapper(is_causal=True)  ──► 加载 Diffusion 模型
│   │   ├─ WanTextEncoder()                      ──► 加载 T5 + Tokenizer
│   │   └─ WanVAEWrapper()                       ──► 加载 VAE
│   │
│   └─ utils/wan_wrapper.py:127-128
│       └─ CausalWanModel.from_pretrained("wan_models/Wan2.1-T2V-1.3B/")
│           ├─ config.json              (模型配置)
│           └─ diffusion_pytorch_model.safetensors (基础权重)
│
└─[52-63] Checkpoint 加载判断
    ├─ --no_checkpoint     ──► 跳过，保持基础权重
    └─ 有 checkpoint       ──► load_state_dict() 覆盖 generator
        ├─ --use_ema       ──► generator_ema
        └─ 无 --use_ema    ──► generator
```
