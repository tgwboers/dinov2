#!/bin/bash

### SETUP DIRECTORY TO WORK IN ###
cd /script

if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo ".env file not found!"
    exit 1
fi

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_DIR=/script
export WANDB_CONFIG_DIR=/script
export WANDB_CACHE_DIR=/script
export WANDB_START_METHOD="thread"
export HOME=/script
export WANDB_DISABLED=$WANDB_DISABLED 
wandb login

CONFIG_FILE=$1
OUTPUT_DIR=$2

### RUN DINOV2 ON GASTRONET ###
torchrun --nnodes 1 --nproc_per_node 2 -m dinov2.train.train \
    --config-file $CONFIG_FILE \
    --output-dir  $OUTPUT_DIR \
    train.dataset_path=GastroNet5M:root=/data:extra=/data