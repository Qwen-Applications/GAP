#!/bin/bash

QWEN3_MODEL_PATH=${QWEN3_MODEL_PATH:-"/root/model/Qwen3-VL-235B-Instruct"}
export VLLM_RPC_TIMEOUT=3000
python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN3_MODEL_PATH \
    --port 8000 \
    --trust-remote-code \
    --served-model-name qwen3-vl-235b \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --max-num-seqs 2