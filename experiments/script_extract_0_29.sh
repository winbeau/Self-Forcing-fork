#!/bin/bash

# 默认提示词
PROMPT="${1:-A majestic eagle soaring through a cloudy sky, cinematic lighting}"

echo "使用提示词: $PROMPT"
echo ""

OUTPUT="cache/locomotive_rushing"

mkdir -p "$OUTPUT"

# 使用 GPU 1, 2, 3 并行处理 30 层（不使用 GPU 0）
GPUS=(1 2 3)
NUM_GPUS=${#GPUS[@]}

# 为每个 GPU 启动一个后台进程处理分配的层
for gpu_idx in "${!GPUS[@]}"; do
  GPU_ID=${GPUS[$gpu_idx]}
  (
    for i in $(seq $gpu_idx $NUM_GPUS 29); do
      echo "[GPU $GPU_ID] 正在执行层级: $i ..."
      CUDA_VISIBLE_DEVICES=$GPU_ID python experiments/run_extraction_each.py \
        --layer_index "$i" \
        --output_path "$OUTPUT/layer${i}.pt" \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --prompt "$PROMPT"
    done
  ) &
done

# 等待所有后台进程完成
wait

echo ""
echo "所有层级处理完成!"
