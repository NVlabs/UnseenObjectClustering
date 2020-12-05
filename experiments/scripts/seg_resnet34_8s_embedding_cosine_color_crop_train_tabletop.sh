#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --dataset tabletop_object_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_crop_tabletop.yml \
  --solver adam \
  --epochs 16
