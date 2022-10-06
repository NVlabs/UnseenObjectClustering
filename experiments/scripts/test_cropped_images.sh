#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

./tools/test_images.py \
  --imgdir data/hand-object/original_depth_images/ \
  --depth depth_*.png \
  --network seg_resnet34_8s_embedding \
  --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_depth_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_depth_tabletop.yml
