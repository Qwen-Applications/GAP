#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/model/Qwen2.5-VL-7B-Instruct}"
TRAIN_META_FILEPATHS="${TRAIN_META_FILEPATHS:-/root/data/train.jsonl}"
DEFAULT_IMAGE_ROOT_DIRS='{"inhouse_interleave": "", "monet_CogCoM": "/root/data", "monet_ReFocus": "/root/data", "monet_Visual_CoT": "/root/data", "monet_Zebra_CoT_count": "/root/data", "monet_Zebra_CoT_geometry": "/root/data", "monet_Zebra_CoT_visual_search": "/root/data", "geo170k_qa_tuning": "/root/data", "mathv360k": "/root/data", "GEOQA": "/root/data", "MM_Math": "/root/data", "MathVision": "/root/data", "chart": "/root/data", "geo_3k": "/root/data"}'
IMAGE_ROOT_DIRS="${IMAGE_ROOT_DIRS:-$DEFAULT_IMAGE_ROOT_DIRS}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/output/}"
VARIANCE_THRESHOLD="${VARIANCE_THRESHOLD:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
MEMMAP_DTYPE="${MEMMAP_DTYPE:-float32}"
SOURCE="${SOURCE:-target}"
FIXED_VISION_TOKENS="${FIXED_VISION_TOKENS:-36}"
LOG_EVERY="${LOG_EVERY:-50}"
COMPUTE_RECONSTRUCTION_LOSS="${COMPUTE_RECONSTRUCTION_LOSS:-1}"

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
    echo "MODEL_NAME_OR_PATH is required"
    exit 1
fi

if [[ -z "${TRAIN_META_FILEPATHS}" ]]; then
    echo "TRAIN_META_FILEPATHS is required"
    exit 1
fi


CMD=(
    python3 -m analyze.export_vit_pca
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --train-meta-filepaths "${TRAIN_META_FILEPATHS}"
    --image-root-dirs "$IMAGE_ROOT_DIRS"
    --output-dir "${OUTPUT_DIR}"
    --variance-threshold "${VARIANCE_THRESHOLD}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --torch-dtype "${TORCH_DTYPE}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --chunk-size "${CHUNK_SIZE}"
    --memmap-dtype "${MEMMAP_DTYPE}"
    --source "${SOURCE}"
    --fixed-vision-tokens "${FIXED_VISION_TOKENS}"
    --log-every "${LOG_EVERY}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${COMPUTE_RECONSTRUCTION_LOSS}" == "1" ]]; then
    CMD+=(--compute-reconstruction-loss)
fi

echo "[run_export_vit_pca] ${CMD[*]}"
exec "${CMD[@]}"
