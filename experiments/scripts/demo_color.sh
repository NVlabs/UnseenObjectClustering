#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

./tools/test_images.py \
  --imgdir data/1216T114244 \
  --color color-*.jpg \
  --depth depth-*.png \
  --network seg_resnet34_8s_embedding \
  --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml

./tools/test_images.py \
  --imgdir data/demo \
  --color *-color.png \
  --depth *-depth.png \
  --network seg_resnet34_8s_embedding \
  --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml
