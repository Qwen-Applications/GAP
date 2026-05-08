#!/bin/bash

set -e

cd /root/code/Gap
python3 -m data.split_data \
    --meta_path /root/data/stage1_data/metadata.json \
    --output_path /root/data/stage1_data/ \
    --val_ratio 0.1 \
    --seed 12345