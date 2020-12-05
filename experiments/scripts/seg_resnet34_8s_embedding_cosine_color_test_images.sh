#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir1="output/shapenet_object/shapenet_object_train"

time ./tools/test_segmentation.py --gpu $1 \
  --imgdir data/images \
  --color *.jpg \
  --network seg_resnet34_8s_embedding \
  --pretrained $outdir1/seg_resnet34_8s_embedding_multi_epoch_16.checkpoint.pth  \
  --pretrained_rrn $outdir1/seg_rrn_unet_epoch_15.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml \
