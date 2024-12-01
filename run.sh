#!/bin/bash

if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo ".env file not found!"
    exit 1
fi

CONFIG=dinov2/configs/train/vitg14_cosmo2.yaml
OUTPUT_DIR="dinov2/output/Experiment 1"

cd $HOME

docker run -it --user 1000:1000 --shm-size=32G --gpus=all -v $HOME:/script -v $DATA:/data tgwboers/dinov2:v1 /script/run_training.sh "$CONFIG" "$OUTPUT_DIR"