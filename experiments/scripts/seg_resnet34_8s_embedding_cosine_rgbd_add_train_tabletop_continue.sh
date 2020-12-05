#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/tabletop_object/tabletop_object_train/seg_resnet34_8s_embedding_cosine_rgbd_sampling_epoch_14.checkpoint.pth \
  --startepoch 14 \
  --dataset tabletop_object_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
  --solver adam \
  --epochs 16
