
set -e

workdir='.'
model_name='vggt'
model_name='page6'
model_weights="/workspace/data/kaichen/log_ckpts/final_7_gra/ckpts/checkpoint.pt"
datasets=('sintel' 'bonn' 'dyncheck')
num_mask=0

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    mkdir -p "$output_dir" 
    CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 4  launch_page.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518 \
        --num_mask "$num_mask"
    CUDA_VISIBLE_DEVICES=3 python3 eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done

