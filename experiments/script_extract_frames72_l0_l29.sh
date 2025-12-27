#!/bin/bash

# ============================================================
# 72帧注意力提取脚本（带安全终止机制）
#
# Ctrl+C 可以安全终止所有进程
# ============================================================

set -e

# 默认提示词
PROMPT="${1:-A majestic eagle soaring through a cloudy sky, cinematic lighting}"

# 输出目录（可通过第二个参数指定）
OUTPUT="${2:-cache/frames72/eagle_soaring}"

echo "使用提示词: $PROMPT"
echo "输出目录: $OUTPUT"
echo ""

# GPU 配置
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# 记录所有子进程 PID
CHILD_PIDS=()

# 清理函数：终止所有子进程
cleanup() {
    echo ""
    echo "=========================================="
    echo "收到终止信号，正在清理所有进程..."
    echo "=========================================="

    # 1. 终止所有记录的子进程
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止进程组 $pid ..."
            kill -TERM -"$pid" 2>/dev/null || true
        fi
    done

    sleep 1

    # 2. 强制杀死所有相关的 python 进程
    echo "强制终止所有 run_extraction_streaming.py 进程..."
    pkill -9 -f "run_extraction_streaming.py" 2>/dev/null || true

    # 3. 杀死 torch compile worker
    pkill -9 -f "compile_worker" 2>/dev/null || true

    sleep 2

    # 4. 检查 GPU 状态
    echo ""
    echo "GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

    echo ""
    echo "清理完成!"
    exit 1
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "72帧注意力提取 (30层并行)"
echo "=========================================="
echo "提示词: $PROMPT"
echo "输出目录: $OUTPUT"
echo "GPU: ${GPUS[*]}"
echo ""
echo "按 Ctrl+C 可以安全终止所有进程"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT"

# 为每个 GPU 启动一个后台进程处理分配的层
for gpu_idx in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$gpu_idx]}

    # 使用 setsid 创建新的进程组，便于统一终止
    setsid bash -c "
        for i in \$(seq $gpu_idx $NUM_GPUS 29); do
            echo \"[GPU $GPU_ID] 正在执行层级: \$i ...\"
            CUDA_VISIBLE_DEVICES=$GPU_ID python experiments/run_extraction_streaming.py \
                --layer_index \"\$i\" \
                --num_frames 72 \
                --output_path \"$OUTPUT/layer\${i}.pt\" \
                --checkpoint_path checkpoints/self_forcing_dmd.pt \
                --prompt \"$PROMPT\"

            if [ \$? -ne 0 ]; then
                echo \"[GPU $GPU_ID] 层级 \$i 失败!\"
            else
                echo \"[GPU $GPU_ID] 层级 \$i 完成!\"
            fi
        done
    " &

    # 记录进程组 leader 的 PID
    CHILD_PIDS+=($!)
    echo "启动 GPU $GPU_ID 进程组 (PID: $!)"
done

echo ""
echo "所有 GPU 进程已启动，等待完成..."
echo "子进程 PIDs: ${CHILD_PIDS[*]}"
echo ""

# 等待所有后台进程完成
wait

echo ""
echo "=========================================="
echo "所有层级处理完成!"
echo "=========================================="
echo "输出目录: $OUTPUT"
ls -la "$OUTPUT"/*.pt 2>/dev/null | wc -l | xargs -I {} echo "已生成 {} 个文件"
