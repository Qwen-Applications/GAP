#!/bin/bash

# === Distributed environment ===
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo "   RANK: $RANK"
echo "   WORLD_SIZE: $WORLD_SIZE"

# === Environment variables ===
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_IB_CUDA_SUPPORT=0
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_SHM_DISABLE=0
# export NCCL_DEBUG=INFO

# === CUDA/NCCL stability settings ===
# Reduce CUDA memory fragmentation and lower OOM risk.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Let NCCL propagate the real asynchronous CUDA error.
export NCCL_ASYNC_ERROR_HANDLING=1

# === Paths ===
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Model path (Qwen2.5-VL)
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen2.5-VL-7B-Instruct"}

# DeepSpeed config
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"$REPO_ROOT/deepspeed/ds_config.json"}

# Data paths
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/root/data/stage1_subset_train.jsonl"}
VAL_DATA_PATH=${VAL_DATA_PATH:-"/root/data/stage1_subset_val.jsonl"}

DEFAULT_IMAGE_ROOTS='{"inhouse_interleave": "", "monet_CogCoM": "/root/data", "monet_ReFocus": "/root/data", "monet_Visual_CoT": "/root/data", "monet_Zebra_CoT_count": "/root/data", "monet_Zebra_CoT_geometry": "/root/data", "monet_Zebra_CoT_visual_search": "/root/data", "geo170k_qa_tuning": "/root/data", "mathv360k": "/root/data", "GEOQA": "/root/data", "MM_Math": "/root/data", "MathVision": "/root/data", "chart": "/root/data", "geo_3k": "/root/data"}'
IMAGE_ROOTS=${IMAGE_ROOTS:-"$DEFAULT_IMAGE_ROOTS"}

# Output paths
OUTPUT_DIR="${PRIMUS_SAVE_CHECKPOINT_DIR:-/root/output/gap_stage1_train}"
LOGGING_DIR="${PRIMUS_TENSORBOARD_LOG_DIR:-/root/output/gap_stage1_logs}"

# === Training hyperparameters ===
PER_DEVICE_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$(( GLOBAL_BATCH_SIZE / (NPROC_PER_NODE * WORLD_SIZE) ))}
LR=${LR:-1e-5}
VISION_LR=${VISION_LR:-1e-5}
EPOCHS=${EPOCHS:-2}
LATENT_SEQ_LEN=${LATENT_SEQ_LEN:-36}
FREEZE_LLM=${FREEZE_LLM:-false}
FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-false}
FREEZE_EMBED=${FREEZE_EMBED:-false}
LATENT_LOSS_WEIGHT=${LATENT_LOSS_WEIGHT:-1.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-32768}
IS_TRANSFORMER_BLOCK=${IS_TRANSFORMER_BLOCK:-false}
NUM_TRANSFORMER_LAYERS=${NUM_TRANSFORMER_LAYERS:-4}
USE_SINKHORN=${USE_SINKHORN:-false}
NO_LATENT_LOSS=${NO_LATENT_LOSS:-false}
PARSER_START_TOKEN=${PARSER_START_TOKEN:-"<parser>"}
PARSER_END_TOKEN=${PARSER_END_TOKEN:-"</parser>"}
PARSE_PROMPT=${PARSE_PROMPT:-"\nPlease parse latent information from the above conversation and provide it in a structured format between <parser> and </parser> tags. The latent information should be concise and directly related to the visual content of the conversation."}
NO_SPLIT=${NO_SPLIT:-false}
CASE_PROBS=${CASE_PROBS:-"1,0,0"}
ACC_THRESHOLD=${ACC_THRESHOLD:-0.0}
VISION_PCA_MU_PATH=${VISION_PCA_MU_PATH:-""}
VISION_PCA_W_PATH=${VISION_PCA_W_PATH:-""}
VISION_HEAD_MODE=${VISION_HEAD_MODE:-"linear"}
DETACH_VISION_HIDDEN=${DETACH_VISION_HIDDEN:-false}
NO_LATENT=${NO_LATENT:-false}
SCHEDULE_SAMPLING_ENABLED=${SCHEDULE_SAMPLING_ENABLED:-false}
SCHEDULE_SAMPLING_RATIO=${SCHEDULE_SAMPLING_RATIO:-0.0}
SCHEDULE_SAMPLING_MAX_RATIO=${SCHEDULE_SAMPLING_MAX_RATIO:-0.0}
SCHEDULE_SAMPLING_WARMUP_STEPS=${SCHEDULE_SAMPLING_WARMUP_STEPS:-0}

VISION_PCA_ARGS=()
if [ -n "$VISION_PCA_MU_PATH" ]; then
    VISION_PCA_ARGS+=(--vision_pca_mu_path "$VISION_PCA_MU_PATH")
fi
if [ -n "$VISION_PCA_W_PATH" ]; then
    VISION_PCA_ARGS+=(--vision_pca_W_path "$VISION_PCA_W_PATH")
fi

echo "=== Start Training ==="

python3 -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    -m training.train \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --vision_feature_size 3584 \
    --vision_head_mode $VISION_HEAD_MODE \
    --detach_vision_hidden $DETACH_VISION_HIDDEN \
    --no_latent $NO_LATENT \
    --latent_seq_len $LATENT_SEQ_LEN \
    "${VISION_PCA_ARGS[@]}" \
    --latent_start_token "<|latent_start|>" \
    --latent_end_token "<|latent_end|>" \
    --latent_pad_token "<|latent_pad|>" \
    --parser_start_token $PARSER_START_TOKEN \
    --parser_end_token $PARSER_END_TOKEN \
    --parse_prompt "$PARSE_PROMPT" \
    --latent_loss_weight $LATENT_LOSS_WEIGHT \
    --is_transformer_block $IS_TRANSFORMER_BLOCK \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --freeze_llm $FREEZE_LLM \
    --freeze_lm_head $FREEZE_LM_HEAD \
    --freeze_embed $FREEZE_EMBED \
    --train_meta_filepath $TRAIN_DATA_PATH \
    --val_meta_filepath $VAL_DATA_PATH \
    --image_root_dirs "$IMAGE_ROOTS" \
    --data_max_seq_len $MAX_SEQ_LEN \
    --no_latent_loss $NO_LATENT_LOSS \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --use_sinkhorn_loss $USE_SINKHORN \
    --schedule_sampling_enabled $SCHEDULE_SAMPLING_ENABLED \
    --schedule_sampling_ratio $SCHEDULE_SAMPLING_RATIO \
    --schedule_sampling_max_ratio $SCHEDULE_SAMPLING_MAX_RATIO \
    --schedule_sampling_warmup_steps $SCHEDULE_SAMPLING_WARMUP_STEPS \
    --learning_rate $LR \
    --vision_head_lr $VISION_LR \
    --lr_scheduler_type "cosine" \
    --num_train_epochs $EPOCHS \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --do_eval \
    --bf16 True \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_num_workers 16 \
    --seed 12345 \
    --save_only_model True \
    --no_split $NO_SPLIT \
    --case_probs $CASE_PROBS \
    --acc_threshold $ACC_THRESHOLD

TRAIN_EXIT=$?
echo "=== Training exit code: $TRAIN_EXIT ==="

# Auto-generate summary on rank 0 of the master node, regardless of train exit code
# (a partial summary on a crashed run is more useful than no summary).
if [ "${RANK:-0}" = "0" ]; then
    SUMMARY_OSS_DIR=${PRIMUS_OUTPUT_DIR:-/primus_oss/output}
    SUMMARY_RUN_NAME="$(basename "$OUTPUT_DIR")_$(date +%Y%m%d_%H%M%S)"
    SUMMARY_LOCAL_DIR="/root/output/training_summary_${SUMMARY_RUN_NAME}"

    mkdir -p "$SUMMARY_OSS_DIR"

    echo "=== Generating training summary -> $SUMMARY_LOCAL_DIR ==="
    python3 "$REPO_ROOT/scripts/summarize_training.py" \
        "$OUTPUT_DIR" \
        --logging-dir "$LOGGING_DIR" \
        --out "$SUMMARY_LOCAL_DIR" \
        || echo "[warn] summarize_training.py failed (continuing)"

    if [ -d "$SUMMARY_LOCAL_DIR" ]; then
        ZIP_NAME="training_summary_${SUMMARY_RUN_NAME}.zip"
        (cd "$(dirname "$SUMMARY_LOCAL_DIR")" && zip -rq "$ZIP_NAME" "$(basename "$SUMMARY_LOCAL_DIR")")
        cp "$(dirname "$SUMMARY_LOCAL_DIR")/$ZIP_NAME" "$SUMMARY_OSS_DIR/$ZIP_NAME" \
            && echo "=== Summary archived to $SUMMARY_OSS_DIR/$ZIP_NAME ==="
        # Also copy the markdown alone for quick browsing on OSS
        if [ -f "$SUMMARY_LOCAL_DIR/summary.md" ]; then
            cp "$SUMMARY_LOCAL_DIR/summary.md" "$SUMMARY_OSS_DIR/summary_${SUMMARY_RUN_NAME}.md"
        fi
    else
        echo "[warn] Summary dir not produced; skipping archive."
    fi
fi

exit $TRAIN_EXIT
