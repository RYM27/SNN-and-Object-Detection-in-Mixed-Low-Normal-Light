#!/usr/bin/env bash

#MODEL=$1
nGPUs=$1

python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py \
--data-path "/home/rama/data rama/thesis/switch YOLO/VGG/dataset" \
--output_dir efficientformerv2_l_exdark_coco
