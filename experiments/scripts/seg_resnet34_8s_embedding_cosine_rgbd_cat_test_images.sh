#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_images.py \
  --imgdir data/images \
  --color *-color.png \
  --depth *-depth.png \
  --network seg_resnet34_8s_embedding \
  --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_cat_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_cat_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_cat_tabletop.yml
