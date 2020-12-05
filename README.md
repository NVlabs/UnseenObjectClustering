# Unseen object instance segmentation
Unseen object instance segmentation by learning features for clustering

### Installation

1. Install [PyTorch](https://pytorch.org/).

2. Compile the new layers under $ROOT/lib/layers
    ```Shell
    cd $ROOT/lib/layers
    python3 setup.py install --user
    ```

### Tested environment

- Ubuntu 16.04
- PyTorch 1.4.0
- CUDA 10.2
- Python 3.5


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
