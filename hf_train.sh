#!/bin/bash


export HF_DATASETS_CACHE="data/.hf_dataset_cache"
# BASE_MODEL_DIR=../../models
BASE_MODEL_NAME='' # zephyr-7b #
DATA_ROOT=data

BASE_MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0 # HuggingFaceH4/zephyr-7b-beta # # $BASE_MODEL_DIR/$BASE_MODEL_NAME
# ====== Config ======
# SEQ_LENGTH=512  # Mistral support up to 32768
SEQ_LENGTH=1024  # Mistral support up to 32768

# Learning Rate
# LR=5e-7
LR=3e-5
WEIGHT_DECAY=0.0


# Batch Size
PER_DEVICE_BSZ=8
GRAD_ACC=1

# Steps
# N_EPOCH=10
N_EPOCH=10
RATIO=0.03


echo "Using model from $BASE_MODEL_PATH"
EXP_ID="${BASE_MODEL_PATH}-lr${LR}-seq${SEQ_LENGTH}-ratio${RATIO}"

# - Output Dir
OUTPUT_DIR=models/ckpts/$EXP_ID
export WANDB_PROJECT=SIU
export WANDB_NAME=$EXP_ID
# export WANDB_MODE=disabled  # for debug
echo "Saving to $OUTPUT_DIR"


# python \
# CUDA_VISIBLE_DEVICES="0,1" \
# CUDA_VISIBLE_DEVICES="0" 
accelerate launch \
    --config_file config/fsdp_.yaml \
    --main_process_port 29700 \
    hf_train.py \
    --model_name_or_path $BASE_MODEL_PATH \
    --with_tracking \
    --output_dir $OUTPUT_DIR \
    --report_to wandb \
    --seed 42 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $PER_DEVICE_BSZ \
    --gradient_accumulation_steps $GRAD_ACC \
    --num_train_epochs $N_EPOCH \
    --checkpointing_steps "epoch" \
    --lr_scheduler_type linear \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_warmup_ratio $RATIO \
    --block_size $SEQ_LENGTH \
    --train_file $DATA_ROOT/pretrainDoc_train.json \
    --validation_file $DATA_ROOT/pretrainDoc_dev.json \
    # --checkpointing_steps 1000000 \
    # --no_keep_linebreaks \
    # --resume_from_checkpoint models/ckpts/$EXP_ID/step_800 \
    # --resume_from_checkpoint models/ckpts/$EXP_ID/epoch_4 \
    
    # --config_file config/accelerate.yaml \
    # --checkpointing_steps "epoch" \
