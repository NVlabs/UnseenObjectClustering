#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir1="data/checkpoints"

./ros/test_images_segmentation.py --gpu 0 \
  --network seg_resnet34_8s_embedding \
  --pretrained $outdir1/seg_resnet34_8s_embedding_cosine_rgbd_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop $outdir1/seg_resnet34_8s_embedding_cosine_rgbd_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
