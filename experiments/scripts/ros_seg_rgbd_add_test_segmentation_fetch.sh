#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir="output/tabletop_object/tabletop_object_train"

./ros/test_images_segmentation.py --gpu $1 \
  --network seg_resnet34_8s_embedding \
  --pretrained $outdir/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop $outdir/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
