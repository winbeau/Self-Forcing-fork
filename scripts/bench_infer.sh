#!/bin/bash
set -euo pipefail

# ======== Defaults ========
CONFIG_PATH="configs/self_forcing_dmd.yaml"
CHECKPOINT_PATH=""
DATA_PATH="prompts/MovieGenVideoBench_num32.txt"
OUTPUT_DIR="outputs/movie_gen_bench"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_OUTPUT_FRAMES=21
SEED=0
USE_EMA=false

# ======== Parse arguments ========
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG_PATH="$2";       shift 2 ;;
        --checkpoint)   CHECKPOINT_PATH="$2";   shift 2 ;;
        --data)         DATA_PATH="$2";         shift 2 ;;
        --output)       OUTPUT_DIR="$2";        shift 2 ;;
        --num_gpus)     NUM_GPUS="$2";          shift 2 ;;
        --num_frames)   NUM_OUTPUT_FRAMES="$2"; shift 2 ;;
        --seed)         SEED="$2";              shift 2 ;;
        --use_ema)      USE_EMA=true;           shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ======== 1. Generate prompts.csv ========
echo "[1/3] Generating prompts.csv ..."
{
    echo "index,prompt"
    idx=0
    while IFS= read -r line; do
        escaped="${line//\"/\"\"}"
        printf '%03d,"%s"\n' "$idx" "$escaped"
        idx=$((idx + 1))
    done < "$DATA_PATH"
} > "$OUTPUT_DIR/prompts.csv"
TOTAL_PROMPTS=$idx
echo "  -> $OUTPUT_DIR/prompts.csv  ($TOTAL_PROMPTS prompts)"

# ======== 2. Background rename watcher ========
MODEL_TAG="regular"
$USE_EMA && MODEL_TAG="ema"

rename_videos() {
    while true; do
        for f in "$OUTPUT_DIR"/*-*_${MODEL_TAG}.mp4; do
            [ -f "$f" ] || continue
            base=$(basename "$f")
            raw_idx=${base%%-*}                       # "5" from "5-0_regular.mp4"
            idx=$((10#$raw_idx))                      # strip leading zeros safely
            new_name=$(printf "video_%03d.mp4" "$idx")
            target="$OUTPUT_DIR/$new_name"
            [ -f "$target" ] && continue
            # wait for file to finish writing (size stable for 1 s)
            sz1=$(stat -c%s "$f" 2>/dev/null) || continue
            sleep 1
            sz2=$(stat -c%s "$f" 2>/dev/null) || continue
            [ "$sz1" != "$sz2" ] && continue
            mv -- "$f" "$target"
            echo "  renamed: $base -> $new_name"
        done
        sleep 2
    done
}

rename_videos &
WATCHER_PID=$!
cleanup() { kill "$WATCHER_PID" 2>/dev/null; wait "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[2/3] File watcher started (PID $WATCHER_PID)"

# ======== 3. Launch multi-GPU inference ========
EXTRA_FLAGS=""
$USE_EMA && EXTRA_FLAGS="--use_ema"
[ -n "$CHECKPOINT_PATH" ] && EXTRA_FLAGS="$EXTRA_FLAGS --checkpoint_path $CHECKPOINT_PATH"

echo "[3/3] Launching torchrun on $NUM_GPUS GPU(s) ..."
torchrun --nproc_per_node="$NUM_GPUS" \
    inference.py \
    --config_path  "$CONFIG_PATH" \
    --data_path    "$DATA_PATH" \
    --output_folder "$OUTPUT_DIR" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --seed "$SEED" \
    --save_with_index \
    $EXTRA_FLAGS

# ======== 4. Final rename sweep ========
cleanup
trap - EXIT

for f in "$OUTPUT_DIR"/*-*_${MODEL_TAG}.mp4; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    raw_idx=${base%%-*}
    idx=$((10#$raw_idx))
    new_name=$(printf "video_%03d.mp4" "$idx")
    target="$OUTPUT_DIR/$new_name"
    [ -f "$target" ] && continue
    mv -- "$f" "$target"
    echo "  renamed: $base -> $new_name"
done

# ======== Summary ========
COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'video_*.mp4' | wc -l)
echo ""
echo "Done!  $COUNT / $TOTAL_PROMPTS videos in $OUTPUT_DIR/"
echo "  prompts.csv : $OUTPUT_DIR/prompts.csv"
