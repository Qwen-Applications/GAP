#!/bin/bash

# === Distributed environment ===
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo "   RANK: $RANK"
echo "   WORLD_SIZE: $WORLD_SIZE"

# === Environment variables ===
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# === Paths ===
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen2.5-VL-7B-Instruct"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"$REPO_ROOT/deepspeed/ds_config.json"}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/root/data/stage1_subset_train.jsonl"}
VAL_DATA_PATH=${VAL_DATA_PATH:-"/root/data/stage1_subset_val.jsonl"}
DEFAULT_IMAGE_ROOTS='{"inhouse_interleave": "", "monet_CogCoM": "/root/data", "monet_ReFocus": "/root/data", "monet_Visual_CoT": "/root/data", "monet_Zebra_CoT_count": "/root/data", "monet_Zebra_CoT_geometry": "/root/data", "monet_Zebra_CoT_visual_search": "/root/data"}'
IMAGE_ROOTS=${IMAGE_ROOTS:-"$DEFAULT_IMAGE_ROOTS"}

# === Output paths ===
OUTPUT_DIR="${PRIMUS_OUTPUT_DIR:-/root/output/gap_debug_train}"
LOGGING_DIR="${PRIMUS_TENSORBOARD_LOG_DIR:-/root/output/gap_debug_logs}"

# === Training hyperparameters ===
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LR=${LR:-1e-5}
VISION_LR=${VISION_LR:-1e-5}
EPOCHS=${EPOCHS:-1}
LATENT_SEQ_LEN=${LATENT_SEQ_LEN:-36}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
VISION_HEAD_MODE=${VISION_HEAD_MODE:-"linear"}
ACC_THRESHOLD=${ACC_THRESHOLD:-0.0}
NO_SPLIT=${NO_SPLIT:-false}
CASE_PROBS=${CASE_PROBS:-"1,0,0"}
FREEZE_LLM=${FREEZE_LLM:-false}
FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-false}
FREEZE_EMBED=${FREEZE_EMBED:-false}

echo "=== Starting GAP Debug Training ==="

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
    --latent_seq_len $LATENT_SEQ_LEN \
    --latent_start_token "<|latent_start|>" \
    --latent_end_token "<|latent_end|>" \
    --latent_pad_token "<|latent_pad|>" \
    --latent_loss_weight 1.0 \
    --freeze_llm $FREEZE_LLM \
    --freeze_lm_head $FREEZE_LM_HEAD \
    --freeze_embed $FREEZE_EMBED \
    --train_meta_filepath $TRAIN_DATA_PATH \
    --val_meta_filepath $VAL_DATA_PATH \
    --image_root_dirs "$IMAGE_ROOTS" \
    --data_max_seq_len $MAX_SEQ_LEN \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LR \
    --vision_head_lr $VISION_LR \
    --lr_scheduler_type "cosine" \
    --num_train_epochs $EPOCHS \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --do_eval \
    --bf16 True \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --seed 12345 \
    --dataloader_num_workers 0 \
    --debug_mode True \
    --no_split $NO_SPLIT \
    --case_probs $CASE_PROBS \
    --acc_threshold $ACC_THRESHOLD
