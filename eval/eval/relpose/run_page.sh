#!/bin/bash
# rm -rf eval_output eval_results
# bash eval/relpose/run_vggt.sh

set -e

workdir='.'
model_weights="/workspace/data/kaichen/log_ckpts/final_7_gra/ckpts/checkpoint.pt"
datasets=('sintel' 'tum')
num_mask=0

for data in "${datasets[@]}"; do
    output_dir="${workdir}/${data}_${model_name}"
    echo "$output_dir"
    CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --main_process_port 29558 run_page.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --num_mask "$num_mask"
done


