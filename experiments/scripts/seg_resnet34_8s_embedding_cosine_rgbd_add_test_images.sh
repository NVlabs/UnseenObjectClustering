#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_images.py \
  --imgdir data/images/test_hand_crops/right_hand \
  --color right_rgb_crop_*.jpg \
  --depth right_depth_crop_*.png \
  --network seg_resnet34_8s_embedding \
  --pretrained output/obman/object_manipulation_train/seg_resnet34_8s_embedding_cosine_sampling_hand_epoch_$2.checkpoint.pth  \
  --dataset object_manipulation_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_obman.yml
