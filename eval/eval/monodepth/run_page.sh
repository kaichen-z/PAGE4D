#!/bin/bash
set -e
# bash eval/monodepth/run_ours.sh

model_name='page6'
model_weights="/workspace/data/kaichen/log_ckpts/final_7_gra/ckpts/checkpoint.pt"
datasets=('sintel' 'bonn' 'dyncheck')
num_mask=0

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    echo "$output_dir"
    CUDA_VISIBLE_DEVICES=1 python3 launch_page.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --num_mask "$num_mask"
done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    CUDA_VISIBLE_DEVICES=1 python3 eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done
