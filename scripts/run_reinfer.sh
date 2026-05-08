#!/bin/bash

set -euo pipefail

# === Distributed environment ===
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
echo "========================================"
echo " Re-Inference"
echo "   RANK:       $RANK"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "========================================"

# === Runtime settings ===
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_USE_V1=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export NCCL_WORK_FIFO_DEPTH=4194304
export LANG=C.UTF-8
export LANGUAGE=C.UTF-8

# Cluster-specific settings for t07.
if [ "${PRIMUS_JOB_RESOURCE_POOL_CLUSTER_TAG}" == "L20C/xjp-tre01" ]; then # t07
    cp -fr /primus_platform_internal/packages/pip/pip.conf /usr/
    export PIP_EXTRA_INDEX_URL="https://mirrors.cloud.aliyuncs.com"
    export NCCL_NVLS_ENABLE=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    export PIP_EXTRA_INDEX_URL="https://mirrors.cloud.aliyuncs.com" 
    sed -i 's/pypi.ngc.nvidia.com/mirrors.cloud.aliyuncs.com/g' /usr/pip.conf 
    sed -i 's/mirrors.aliyun.com/mirrors.cloud.aliyuncs.com/g' /usr/pip.conf 
    apt update
fi


INPUT_PATH=${INPUT_PATH:-"/root/data/train.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"/root/data/train_reinfer.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/output"}
IMAGE_ROOT=${IMAGE_ROOT:-"/root/data"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"/root/code/Gap/data/re_infer_prompt_template.txt"}
MODEL_NAME=${MODEL_NAME:-"qwen3-vl-235b"}
API_BASE=${API_BASE:-"http://localhost:8000"}
FEW_SHOTS=${FEW_SHOTS:-"/root/code/Gap/data/few_shot_samples.txt"}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.9}
TIMEOUT=${TIMEOUT:-3600}
MAX_RETRIES=${MAX_RETRIES:-3}
REPLACE_ASSISTANT=${REPLACE_ASSISTANT:-"true"}

# vLLM config
QWEN3_MODEL_PATH=${QWEN3_MODEL_PATH:-"/root/model/Qwen3-VL-235B-Instruct"}
TP_SIZE=${TP_SIZE:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCH_TOKEN=${MAX_NUM_BATCH_TOKEN:-24576}

# Extract port from API_BASE.
VLLM_PORT=$(echo "$API_BASE" | grep -oE '[0-9]+')

# === Step 0: Start vLLM server with watchdog ===
VLLM_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$VLLM_LOG_DIR"
export VLLM_RPC_TIMEOUT=3000

# Helper for starting vLLM.
start_vllm() {
    echo "[vLLM] Starting server (TP=$TP_SIZE, port=$VLLM_PORT)..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$QWEN3_MODEL_PATH" \
        --port "$VLLM_PORT" \
        --trust-remote-code \
        --served-model-name "$MODEL_NAME" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCH_TOKEN" \
        >> "$VLLM_LOG_DIR/vllm_rank${RANK}.log" 2>&1 &
    VLLM_PID=$!
    echo "[vLLM] PID: $VLLM_PID, log: $VLLM_LOG_DIR/vllm_rank${RANK}.log"
}

# Watchdog: restart vLLM if it exits.
VLLM_WATCHDOG_STOP="/tmp/vllm_watchdog_stop_rank${RANK}"
rm -f "$VLLM_WATCHDOG_STOP"

vllm_watchdog() {
    while [[ ! -f "$VLLM_WATCHDOG_STOP" ]]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[vLLM-Watchdog] Server (PID=$VLLM_PID) died, restarting..." >&2
            sleep 5
            start_vllm
        fi
        sleep 10
    done
}

# Initial start.
start_vllm
# Start watchdog in the background.
vllm_watchdog &
WATCHDOG_PID=$!
echo "[vLLM-Watchdog] PID: $WATCHDOG_PID"

# === Step 1: Split data ===
echo "[Step 1] Splitting data (shard $RANK / $WORLD_SIZE)..."
SHARD_DIR="${OUTPUT_DIR}/shards"
SHARD_INPUT="${SHARD_DIR}/shard_${RANK}.jsonl"
SHARD_OUTPUT="${SHARD_DIR}/output_${RANK}.jsonl"

python3 -m data.shard_data split \
    --input_path "$INPUT_PATH" \
    --num_shards "$WORLD_SIZE" \
    --output_dir "$SHARD_DIR" \
    --rank "$RANK"

# === Step 2: Run inference ===
echo "[Step 2] Running inference on shard $RANK..."
EXTRA_ARGS=""
[[ -n "$FEW_SHOTS" ]] && EXTRA_ARGS="$EXTRA_ARGS --few_shots $FEW_SHOTS"
[[ "$REPLACE_ASSISTANT" == "true" ]] && EXTRA_ARGS="$EXTRA_ARGS --replace_assistant"

python3 -m data.re_infer \
    --input_path "$SHARD_INPUT" \
    --output_path "$SHARD_OUTPUT" \
    --image_root "$IMAGE_ROOT" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --model "$MODEL_NAME" \
    --api_base "$API_BASE" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --timeout "$TIMEOUT" \
    --max_retries "$MAX_RETRIES" \
    $EXTRA_ARGS

echo "[Info] Rank $RANK inference done: $SHARD_OUTPUT"

# === Step 3: Merge outputs on rank 0 only ===
if [[ "$RANK" -eq 0 && "$WORLD_SIZE" -gt 1 ]]; then
    echo "[Step 3] Rank 0: Waiting for all shards and merging..."
    python3 -m data.shard_data merge \
        --input_dir "$SHARD_DIR" \
        --output_path "$OUTPUT_PATH" \
        --num_shards "$WORLD_SIZE" \
        --max_wait 86400
    
    cp "$OUTPUT_PATH" "${PRIMUS_OUTPUT_DIR}/reinfer_data.jsonl"
    echo "[Info] Merged output saved to $OUTPUT_PATH and ${PRIMUS_OUTPUT_DIR}/reinfer_data.jsonl"

elif [[ "$WORLD_SIZE" -eq 1 ]]; then
    cp "$OUTPUT_PATH" "${PRIMUS_OUTPUT_DIR}/reinfer_data.jsonl"
    echo "[Info] Single node mode, output: $OUTPUT_PATH, also copied to ${PRIMUS_OUTPUT_DIR}/reinfer_data.jsonl"
fi

echo "[Done] Rank $RANK finished."
