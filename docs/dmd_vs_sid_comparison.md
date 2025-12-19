# DMD vs SID 对比分析

## 概述

DMD 和 SID 是两种不同的蒸馏方法，用于将多步扩散模型蒸馏为少步生成模型。

---

## 核心区别

| 方面 | DMD (Distribution Matching Distillation) | SID (Score Identity Distillation) |
|------|------------------------------------------|-----------------------------------|
| **论文** | [arXiv:2311.18828](https://arxiv.org/abs/2311.18828) | Score Identity Distillation |
| **代码文件** | `model/dmd.py` | `model/sid.py` |
| **核心思想** | 最小化生成分布与真实分布的 KL 散度 | 利用 Score Identity 构造损失 |
| **需要 Critic** | ✅ 需要训练 fake_score 网络 | ✅ 需要训练 fake_score 网络 |

---

## 损失函数对比

### DMD 损失 (`model/dmd.py:54-126`)

```python
# 计算 KL 梯度 (论文 eq. 7)
grad = pred_fake_image - pred_real_image

# 梯度归一化 (论文 eq. 8)
p_real = estimated_clean_x0 - pred_real_image
normalizer = |p_real|.mean()
grad = grad / normalizer

# 最终损失
dmd_loss = 0.5 * MSE(x0, (x0 - grad).detach())
```

### SID 损失 (`model/sid.py:128-138`)

```python
# Score Identity 损失
sid_loss = (pred_real - pred_fake) * (
    (pred_real - x0) - alpha * (pred_real - pred_fake)
)

# 归一化
normalizer = |x0 - pred_real|.mean()
sid_loss = sid_loss / normalizer
sid_loss = sid_loss.mean()
```

---

## 配置参数差异

| 参数 | DMD (`self_forcing_dmd.yaml`) | SID (`self_forcing_sid.yaml`) |
|------|-------------------------------|-------------------------------|
| `distribution_loss` | `dmd` | `dmd` (注: 配置文件可能有误) |
| `lr_critic` | `4.0e-07` | `2.0e-06` |
| `real_name` | `Wan2.1-T2V-14B` | `Wan2.1-T2V-1.3B` |
| `sid_alpha` | ❌ 无 | `1.0` |
| `dfake_gen_update_ratio` | `5` | `5` |

---

## 训练流程差异

### DMD 训练流程

```
┌─────────────────────────────────────────────────────────┐
│  1. Generator 生成 fake video                           │
│  2. 加噪到 timestep t                                   │
│  3. fake_score 预测 → pred_fake                        │
│  4. real_score 预测 (CFG) → pred_real                  │
│  5. KL grad = pred_fake - pred_real                    │
│  6. 归一化后反向传播更新 Generator                      │
└─────────────────────────────────────────────────────────┘
```

### SID 训练流程

```
┌─────────────────────────────────────────────────────────┐
│  1. Generator 生成 fake video (x0)                      │
│  2. 加噪到 timestep t                                   │
│  3. fake_score 预测 → pred_fake                        │
│  4. real_score 预测 (CFG) → pred_real                  │
│  5. SID loss = (pred_real - pred_fake) *               │
│                ((pred_real - x0) - α*(pred_real-pred_fake)) │
│  6. 归一化后反向传播更新 Generator                      │
└─────────────────────────────────────────────────────────┘
```

---

## 数学公式对比

### DMD 公式

$$\nabla_\theta \mathcal{L}_{DMD} = \mathbb{E}_{t,\epsilon} \left[ \frac{s_\phi(x_t) - s_{real}(x_t)}{|x_0 - s_{real}(x_t)|} \right]$$

### SID 公式

$$\mathcal{L}_{SID} = (s_{real} - s_\phi) \cdot \left[ (s_{real} - x_0) - \alpha (s_{real} - s_\phi) \right]$$

### 符号说明

- $s_\phi$ = fake_score 预测
- $s_{real}$ = real_score 预测 (带 CFG)
- $x_0$ = generator 生成的 clean latent
- $\alpha$ = SID 超参数 (默认 1.0)

---

## 实际效果差异

| 指标 | DMD | SID |
|------|-----|-----|
| 训练稳定性 | 较稳定 | 需要调 alpha |
| 收敛速度 | 较慢 | 较快 |
| 显存占用 | 较低 | 相近 |
| 生成质量 | 基准 | 可能更好 |

---

## 代码结构对比

### DMD 类结构 (`model/dmd.py`)

```python
class DMD(SelfForcingModel):
    def __init__(self, args, device):
        # 初始化 generator, fake_score, real_score
        # 设置 guidance_scale, timestep_shift 等超参数

    def _compute_kl_grad(self, noisy_x, clean_x, timestep, cond, uncond):
        # 计算 KL 散度梯度
        # 返回归一化后的梯度

    def compute_distribution_matching_loss(self, x, cond, uncond, ...):
        # 采样 timestep 和噪声
        # 调用 _compute_kl_grad
        # 返回 MSE 损失

    def generator_loss(self, ...):
        # 运行 generator
        # 计算 DMD 损失

    def critic_loss(self, ...):
        # 运行 generator (no grad)
        # 训练 fake_score 的 denoising loss
```

### SID 类结构 (`model/sid.py`)

```python
class SiD(SelfForcingModel):
    def __init__(self, args, device):
        # 初始化 generator, fake_score, real_score
        # 额外参数: sid_alpha

    def compute_distribution_matching_loss(self, x, cond, uncond, ...):
        # 采样 timestep 和噪声
        # 计算 pred_fake 和 pred_real
        # 计算 SID loss (不同于 DMD)
        # 返回归一化后的损失

    def generator_loss(self, ...):
        # 运行 generator
        # 计算 SID 损失

    def critic_loss(self, ...):
        # 运行 generator (no grad)
        # 训练 fake_score 的 denoising loss
```

---

## 使用命令

### 使用 DMD 训练

```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd
```

### 使用 SID 训练

```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  train.py \
  --config_path configs/self_forcing_sid.yaml \
  --logdir logs/self_forcing_sid
```

### 使用 DMD checkpoint 推理

```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --use_ema
```

### 使用 SID checkpoint 推理

```bash
python inference.py \
    --config_path configs/self_forcing_sid.yaml \
    --checkpoint_path checkpoints/self_forcing_sid.pt \
    --use_ema
```
