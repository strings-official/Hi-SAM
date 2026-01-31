#!/bin/bash

dataset_path=$1
dilation_factor=$2
input_dir=${dataset_path}/images
output_dir=${dataset_path}/hisam_jsons

checkpoint="pretrained_checkpoint/hi_sam_l.pth"
model_type="vit_l"

python demo_amg.py --checkpoint "$checkpoint" --model-type "$model_type" --input "$input_dir" --eval --eval_out_file "$output_dir"
python get_masks.py --dataset_path "$dataset_path" --dilation ${dilation_factor}