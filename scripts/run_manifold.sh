#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

DATA_FILE="${DATA_FILE:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/outputs/manifold}"

if [[ -z "${DATA_FILE}" ]]; then
	echo "[Error] DATA_FILE is required."
	echo "Example: DATA_FILE=/path/to/data.jsonl NPROC_PER_NODE=4 bash ${BASH_SOURCE[0]}"
	exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
	"${PYTHON_BIN}" -m torch.distributed.run
	--nproc_per_node="${NPROC_PER_NODE}"
	--master_port="${MASTER_PORT}"
	-m data.visual_manifold
	--data-file "${DATA_FILE}"
	--image-root "${IMAGE_ROOT}"
	--sample-size "${SAMPLE_SIZE}"
	--seed "${SEED}"
	--output-dir "${OUTPUT_DIR}"
)

echo "[Info] project root: ${PROJECT_ROOT}"
echo "[Info] python bin: ${PYTHON_BIN}"
echo "[Info] nproc_per_node: ${NPROC_PER_NODE}"
echo "[Info] data file: ${DATA_FILE}"
echo "[Info] image root: ${IMAGE_ROOT:-<empty>}"
echo "[Info] sample size: ${SAMPLE_SIZE}"
echo "[Info] seed: ${SEED}"
echo "[Info] output dir: ${OUTPUT_DIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	echo "[Info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

"${CMD[@]}"
