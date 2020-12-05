# Unseen Object Clustering: Learning RGB-D Feature Embeddings for Unseen Object Instance Segmentation

### Introduction

In this work, we propose a new method for unseen object instance segmentation by learning RGB-D feature embeddings from synthetic data. A metric learning loss functionis utilized to learn to produce pixel-wise feature embeddings such that pixels from the same object are close to each other and pixels from different objects are separated in the embedding space. With the learned feature embeddings, a mean shift clustering algorithm can be applied to discover and segment unseen objects. We further improve the segmentation accuracy with a new two-stage clustering algorithm. Our method demonstrates that non-photorealistic synthetic RGB and depth images can be used to learn feature embeddings that transfer well to real-world images for unseen object instance segmentation. [arXiv](https://arxiv.org/pdf/2007.15157.pdf), [Talk video](https://youtu.be/pxma-x0BGpU)

<p align="center"><img src="./data/pics/network.png" width="750" height="200"/></p>

### License

Unseen Object Clustering is released under the NVIDIA Source Code License (refer to the LICENSE file for details).

### Citation

If you find Unseen Object Clustering useful in your research, please consider citing:

    @inproceedings{xiang2020learning,
        Author = {Yu Xiang and Christopher Xie and Arsalan Mousavian and Dieter Fox},
        Title = {Learning RGB-D Feature Embeddings for Unseen Object Instance Segmentation},
        booktitle = {Conference on Robot Learning (CoRL)},
        Year = {2020}
    }


### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above


### Installation

1. Install [PyTorch](https://pytorch.org/).

2. Install python packages
   ```Shell
   pip install -r requirement.txt
   ```




### Pretrained models

Download the trained models from [here](https://drive.google.com/file/d/1RACi8kri5Jx557PFAbhJZO8zULfs_c9X/view?usp=sharing), save them to data/checkpoints


### Test on npy files

Run the following script, change the image folder path to where the npy files are
    ```Shell
    ./experiments/scripts/seg_resnet34_8s_embedding_cosine_rgbd_test_npy.sh $GPU_ID
    ```

### Running

1. Download the Shapenet models with uv-mapping from [here](https://drive.google.com/open?id=1_EsVXieKsckckFgClhBixFI4wZe44_KA).

2. Download the texture images from [here](https://drive.google.com/open?id=1vmDNhnr6H5FM2yQQ8pdZArrO5Gw06QMw).

3. Download the coco 2014 images from [here](http://cocodataset.org/#download).

4. Download the Table Top Dataset from [here](https://drive.google.com/file/d/1fqKszKordLrx1801dAnBMAaGdQ1sRbRA/view?usp=sharing).

4. Create symlinks for the datasets
    ```Shell
    cd $ROOT/data
    ln -s $shapenet_data shapenet
    ln -s $texture_data textures
    ln -s $coco_data coco
    ln -s $tabletop_data tabletop
    ```

5. Training and testing scripts
    ```Shell
    cd $ROOT

    # training for feature learning
    ./experiments/scripts/seg_resnet34_8s_embedding_cosine_train_shapenet.sh

    # testing features
    ./experiments/scripts/seg_resnet34_8s_embedding_cosine_test_shapenet.sh $GPU_ID $EPOCH

    # training for region refinement network
    ./experiments/scripts/seg_resnet34_8s_embedding_cosine_test_shapenet.sh

    # testing region refinement network
    ./experiments/scripts/seg_rrn_unet_test_shapenet.sh $GPU_ID $EPOCH

    # testing on real images with both networks
    ./experiments/scripts/seg_resnet34_8s_embedding_cosine_test_images.sh $GPU_ID
    ```

### Running with ROS for realsense
    ```Shell
    # start realsense
    roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera

    # start rviz
    rosrun rviz rviz -d ./ros/segmentation.rviz

    # run segmentation
    ./experiments/scripts/ros_seg_test_segmentation_realsense.sh $GPU_ID
    ```
