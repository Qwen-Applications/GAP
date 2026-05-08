#!/bin/bash

# Switch to the project root.
cd "$(dirname "$0")/.." || exit

mkdir -p /root/output/
MODEL_NAME="/root/model/Qwen3-235B-A22B-Instruct"
INPUT_FILE="/root/data/qwen2p5_reinfer.jsonl"
PRIMUS_OUTPUT_DIR="/root/output/" # Output copy directory.

# Get shard count and rank from common cluster environment variables.
TOTAL_SHARDS=${WORLD_SIZE:-${SLURM_NNODES:-1}}
SHARD_RANK=${RANK:-${SLURM_NODEID:-0}}

# Include shard rank in the output filename to avoid write conflicts.
OUTPUT_FILE="/root/data/qwen2p5_reinfer_w_acc_part${SHARD_RANK}.jsonl"

echo "==========================================="
echo "Starting Verification"
echo "Model: $MODEL_NAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Total Shards: $TOTAL_SHARDS, Shard Rank: $SHARD_RANK"
echo "==========================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port 8000 \
    --trust-remote-code \
    --served-model-name qwen3-235b \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --max-num-seqs 2 &

VLLM_PID=$!

echo "Waiting for vLLM server to start..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "vLLM is not ready yet. Retrying in 10 seconds..."
    sleep 10
done
echo "vLLM server is up and running!"

python3 -m data.verify_answer \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --model_name qwen3-235b \
    --total_shards $TOTAL_SHARDS \
    --shard_rank $SHARD_RANK

kill $VLLM_PID
echo "vLLM server (PID: $VLLM_PID) has been terminated."

cp $OUTPUT_FILE $PRIMUS_OUTPUT_DIR
echo "Done! Output saved to $PRIMUS_OUTPUT_DIR/$(basename $OUTPUT_FILE)"
