import torch
import flash_attn
from flash_attn import flash_attn_func

print(f"✅ Flash Attention 版本: {flash_attn.__version__}")
print(f"✅ PyTorch 版本: {torch.__version__}")
print(f"✅ CUDA 设备: {torch.cuda.get_device_name(0)}")

# 1. 准备测试数据 (必须是 CUDA + float16 或 bfloat16)
# Shape: (Batch_Size, Seq_Len, Num_Heads, Head_Dim)
batch_size = 2
seq_len = 1024
n_heads = 8
head_dim = 64

dtype = torch.float16 # FlashAttn 必须跑在 fp16 或 bf16 下
device = "cuda"

print("\n🚀 正在生成随机 Tensor...")
q = torch.randn((batch_size, seq_len, n_heads, head_dim), device=device, dtype=dtype)
k = torch.randn((batch_size, seq_len, n_heads, head_dim), device=device, dtype=dtype)
v = torch.randn((batch_size, seq_len, n_heads, head_dim), device=device, dtype=dtype)

# 2. 调用 Flash Attention
print("⚡ 正在执行 Flash Attention 计算...")
try:
    # 调用核心函数
    output = flash_attn_func(q, k, v)
    
    # 验证输出形状
    expected_shape = (batch_size, seq_len, n_heads, head_dim)
    assert output.shape == expected_shape, f"形状错误: {output.shape} != {expected_shape}"
    
    print(f"🎉 成功！输出形状: {output.shape}")
    print("✨ Flash Attention 安装完美，可以投入战斗了！")

except Exception as e:
    print(f"\n❌ 出错了: {e}")
    print("这通常是因为显卡架构太旧（需Ampere以上）或 PyTorch/CUDA 版本不匹配。")
