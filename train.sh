#!/bin/bash
# Selects the appropriate config based on the GPU setup and runs training.

if [ "$1" == "3090" ]; then
    echo "Using configuration for 3090..."
    source config/3090.sh
elif [ "$1" == "8xH100" ]; then
    echo "Using configuration for 8xH100..."
    source config/8xH100.sh
else
    echo "Usage: $0 [3090|8xH100]"
    exit 1
fi

torchrun --standalone --nproc_per_node $NPROC src/train.py \
    ${MICRO_BATCH_SIZE:+--micro_batch_size $MICRO_BATCH_SIZE} \
    ${DATASET_DIR:+--dataset_dir $DATASET_DIR} \
    ${CHECKPOINT_DIR:+--checkpoint_dir $CHECKPOINT_DIR} \
    ${EPOCHS:+--epochs $EPOCHS} \
    ${WANDB_PROJECT:+--wandb_project $WANDB_PROJECT}
