#!/bin/bash
set -euo pipefail

detect_num_gpus() {
    local devices=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        devices=$(nvidia-smi -L 2>/dev/null || true)
        if [[ -n "$devices" ]]; then
            printf '%s\n' "$devices" | wc -l
        else
            echo 0
        fi
    else
        echo 0
    fi
}

# ======== Defaults ========
CONFIG_PATH="configs/self_forcing_dmd_long.yaml"
CHECKPOINT_PATH="checkpoints/self_forcing_dmd.pt"
DATA_PATH="prompts/MovieGenVideoBench_num32.txt"
OUTPUT_DIR="outputs/movie_gen_bench"
NUM_GPUS=$(detect_num_gpus)
MASTER_PORT=29501
NUM_OUTPUT_FRAMES=120
SEED=0
USE_EMA=true
PROFILE=false

usage() {
    cat <<'EOF'
Usage: bash scripts/bench_infer.sh [options]

Options:
  --config PATH         Model config path
  --checkpoint PATH     Checkpoint path
  --data PATH           Prompt txt path
  --output PATH         Output directory
  --num_gpus N          Number of GPUs / torchrun processes
  --master_port PORT    torchrun master port (default: 29501)
  --num_frames N        Number of latent output frames
  --seed N              Base random seed
  --use_ema             Use EMA weights (default)
  --no_use_ema          Disable EMA weights
  --profile             Print diffusion/VAE timing breakdown
  --help                Show this help message
EOF
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid $name: $value" >&2
        exit 1
    fi
}

validate_visible_devices() {
    if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
        return
    fi

    local raw_devices="$CUDA_VISIBLE_DEVICES"
    local -a parsed_devices=()
    local device=""
    local visible_count=0

    IFS=',' read -r -a parsed_devices <<< "$raw_devices"
    for device in "${parsed_devices[@]}"; do
        device="${device//[[:space:]]/}"
        [[ -z "$device" ]] && continue
        visible_count=$((visible_count + 1))
    done

    if [[ "$visible_count" -ne "$NUM_GPUS" ]]; then
        echo "CUDA_VISIBLE_DEVICES count ($visible_count) does not match --num_gpus ($NUM_GPUS)." >&2
        echo "  CUDA_VISIBLE_DEVICES=$raw_devices" >&2
        exit 1
    fi
}

# ======== Parse arguments ========
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG_PATH="$2";       shift 2 ;;
        --checkpoint)   CHECKPOINT_PATH="$2";   shift 2 ;;
        --data)         DATA_PATH="$2";         shift 2 ;;
        --output)       OUTPUT_DIR="$2";        shift 2 ;;
        --num_gpus)     NUM_GPUS="$2";          shift 2 ;;
        --master_port)  MASTER_PORT="$2";       shift 2 ;;
        --num_frames)   NUM_OUTPUT_FRAMES="$2"; shift 2 ;;
        --seed)         SEED="$2";              shift 2 ;;
        --use_ema)      USE_EMA=true;           shift ;;
        --no_use_ema)   USE_EMA=false;          shift ;;
        --profile)      PROFILE=true;           shift ;;
        --help)         usage;                  exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

require_positive_int "--num_gpus" "$NUM_GPUS"
require_positive_int "--master_port" "$MASTER_PORT"
require_positive_int "--num_frames" "$NUM_OUTPUT_FRAMES"
validate_visible_devices

mkdir -p "$OUTPUT_DIR"

# ======== 1. Generate prompts.csv ========
echo "[1/3] Generating prompts.csv ..."
{
    echo "index,prompt"
    idx=0
    while IFS= read -r line; do
        escaped="${line//\"/\"\"}"
        printf '%d,"%s"\n' "$idx" "$escaped"
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
echo "[3/3] Launching torchrun on $NUM_GPUS GPU(s) ..."
TORCHRUN_CMD=(
    torchrun
    "--nproc_per_node=$NUM_GPUS"
    "--master_port=$MASTER_PORT"
    inference.py
    --config_path "$CONFIG_PATH"
    --data_path "$DATA_PATH"
    --output_folder "$OUTPUT_DIR"
    --num_output_frames "$NUM_OUTPUT_FRAMES"
    --seed "$SEED"
    --save_with_index
)

$USE_EMA && TORCHRUN_CMD+=(--use_ema)
$PROFILE && TORCHRUN_CMD+=(--profile)
[[ -n "$CHECKPOINT_PATH" ]] && TORCHRUN_CMD+=(--checkpoint_path "$CHECKPOINT_PATH")

printf '  ->'
printf ' %q' "${TORCHRUN_CMD[@]}"
printf '\n'
"${TORCHRUN_CMD[@]}"

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
