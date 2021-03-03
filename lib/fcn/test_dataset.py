# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys, os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation, _vis_features, _vis_minibatch_segmentation_final
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.mean_shift import mean_shift_smart_init
from utils.evaluation import multilabel_metrics
import utils.mask as util_

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def clustering_features(features, num_seeds=100):
    metric = cfg.TRAIN.EMBEDDING_METRIC
    height = features.shape[2]
    width = features.shape[3]
    out_label = torch.zeros((features.shape[0], height, width))

    # mean shift clustering
    kappa = 20
    selected_pixels = []
    for j in range(features.shape[0]):
        X = features[j].view(features.shape[1], -1)
        X = torch.transpose(X, 0, 1)
        cluster_labels, selected_indices = mean_shift_smart_init(X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric=metric)
        out_label[j] = cluster_labels.view(height, width)
        selected_pixels.append(selected_indices)
    return out_label, selected_pixels


def crop_rois(rgb, initial_masks, depth):

    N, H, W = initial_masks.shape
    crop_size = cfg.TRAIN.SYN_CROP_SIZE
    padding_percentage = 0.25

    mask_ids = torch.unique(initial_masks[0])
    if mask_ids[0] == 0:
        mask_ids = mask_ids[1:]
    num = mask_ids.shape[0]
    rgb_crops = torch.zeros((num, 3, crop_size, crop_size), device=cfg.device)
    rois = torch.zeros((num, 4), device=cfg.device)
    mask_crops = torch.zeros((num, crop_size, crop_size), device=cfg.device)
    if depth is not None:
        depth_crops = torch.zeros((num, 3, crop_size, crop_size), device=cfg.device)
    else:
        depth_crops = None

    for index, mask_id in enumerate(mask_ids):
        mask = (initial_masks[0] == mask_id).float() # Shape: [H x W]
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
        x_padding = int(torch.round((x_max - x_min).float() * padding_percentage).item())
        y_padding = int(torch.round((y_max - y_min).float() * padding_percentage).item())

        # pad and be careful of boundaries
        x_min = max(x_min - x_padding, 0)
        x_max = min(x_max + x_padding, W-1)
        y_min = max(y_min - y_padding, 0)
        y_max = min(y_max + y_padding, H-1)
        rois[index, 0] = x_min
        rois[index, 1] = y_min
        rois[index, 2] = x_max
        rois[index, 3] = y_max

        # crop
        rgb_crop = rgb[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]
        if depth is not None:
            depth_crop = depth[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]

        # resize
        new_size = (crop_size, crop_size)
        rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
        rgb_crops[index] = rgb_crop
        mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
        mask_crops[index] = mask_crop
        if depth is not None:
            depth_crop = F.upsample_bilinear(depth_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
            depth_crops[index] = depth_crop

    return rgb_crops, mask_crops, rois, depth_crops


# labels_crop is the clustering labels from the local patch
def match_label_crop(initial_masks, labels_crop, out_label_crop, rois, depth_crop):
    num = labels_crop.shape[0]
    for i in range(num):
        mask_ids = torch.unique(labels_crop[i])
        for index, mask_id in enumerate(mask_ids):
            mask = (labels_crop[i] == mask_id).float()
            overlap = mask * out_label_crop[i]
            percentage = torch.sum(overlap) / torch.sum(mask)
            if percentage < 0.5:
                labels_crop[i][labels_crop[i] == mask_id] = -1

    # sort the local labels
    sorted_ids = []
    for i in range(num):
        if depth_crop is not None:
            if torch.sum(labels_crop[i] > -1) > 0:
                roi_depth = depth_crop[i, 2][labels_crop[i] > -1]
            else:
                roi_depth = depth_crop[i, 2]
            avg_depth = torch.mean(roi_depth[roi_depth > 0])
            sorted_ids.append((i, avg_depth))
        else:
            x_min = rois[i, 0]
            y_min = rois[i, 1]
            x_max = rois[i, 2]
            y_max = rois[i, 3]
            orig_H = y_max - y_min + 1
            orig_W = x_max - x_min + 1
            roi_size = orig_H * orig_W
            sorted_ids.append((i, roi_size))

    sorted_ids = sorted(sorted_ids, key=lambda x : x[1], reverse=True)
    sorted_ids = [x[0] for x in sorted_ids]

    # combine the local labels
    refined_masks = torch.zeros_like(initial_masks).float()
    count = 0
    for index in sorted_ids:

        mask_ids = torch.unique(labels_crop[index])
        if mask_ids[0] == -1:
            mask_ids = mask_ids[1:]

        # mapping
        label_crop = torch.zeros_like(labels_crop[index])
        for mask_id in mask_ids:
            count += 1
            label_crop[labels_crop[index] == mask_id] = count

        # resize back to original size
        x_min = int(rois[index, 0].item())
        y_min = int(rois[index, 1].item())
        x_max = int(rois[index, 2].item())
        y_max = int(rois[index, 3].item())
        orig_H = int(y_max - y_min + 1)
        orig_W = int(x_max - x_min + 1)
        mask = label_crop.unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0, 0]

        # Set refined mask
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        refined_masks[0, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = resized_mask[h_idx, w_idx].cpu()

    return refined_masks, labels_crop


# filter labels on zero depths
def filter_labels_depth(labels, depth, threshold):
    labels_new = labels.clone()
    for i in range(labels.shape[0]):
        label = labels[i]
        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            roi_depth = depth[i, 2][label == mask_id]
            depth_percentage = torch.sum(roi_depth > 0).float() / torch.sum(mask)
            if depth_percentage < threshold:
                labels_new[i][label == mask_id] = 0

    return labels_new


# filter labels inside boxes
def filter_labels(labels, bboxes):
    labels_new = labels.clone()
    height = labels.shape[1]
    width = labels.shape[2]
    for i in range(labels.shape[0]):
        label = labels[i]
        bbox = bboxes[i].numpy()

        bbox_mask = torch.zeros_like(label)
        for j in range(bbox.shape[0]):
            x1 = max(int(bbox[j, 0]), 0)
            y1 = max(int(bbox[j, 1]), 0)
            x2 = min(int(bbox[j, 2]), width-1)
            y2 = min(int(bbox[j, 3]), height-1)
            bbox_mask[y1:y2, x1:x2] = 1

        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            percentage = torch.sum(mask * bbox_mask) / torch.sum(mask)
            if percentage > 0.8:
                labels_new[i][label == mask_id] = 0

    return labels_new


# test a single sample
def test_sample(sample, network, network_crop):

    # construct input
    image = sample['image_color'].cuda()
    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        depth = sample['depth'].cuda()
    else:
        depth = None

    if 'label' in sample:
        label = sample['label'].cuda()
    else:
        label = None

    # run network
    features = network(image, label, depth).detach()
    out_label, selected_pixels = clustering_features(features, num_seeds=100)

    if depth is not None:
        # filter labels on zero depth
        out_label = filter_labels_depth(out_label, depth, 0.8)

    # zoom in refinement
    out_label_refined = None
    if network_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
            labels_crop, selected_pixels_crop = clustering_features(features_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

    if cfg.TEST.VISUALIZE:
        bbox = None
        _vis_minibatch_segmentation_final(image, depth, label, out_label, out_label_refined, features, 
            selected_pixels=selected_pixels, bbox=bbox)
    return out_label, out_label_refined


# test a dataset
def test_segnet(test_loader, network, output_dir, network_crop):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)

    # switch to test mode
    network.eval()
    if network_crop is not None:
        network_crop.eval()

    metrics_all = []
    metrics_all_refined = []
    for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        image = sample['image_color'].cuda()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth = sample['depth'].cuda()
        else:
            depth = None
        label = sample['label'].cuda()

        # run network
        features = network(image, label, depth).detach()
        out_label, selected_pixels = clustering_features(features, num_seeds=100)

        if 'ocid' in test_loader.dataset.name and depth is not None:
            # filter labels on zero depth
            out_label = filter_labels_depth(out_label, depth, 0.5)

        if 'osd' in test_loader.dataset.name and depth is not None:
            # filter labels on zero depth
            out_label = filter_labels_depth(out_label, depth, 0.8)

        # evaluation
        gt = sample['label'].squeeze().numpy()
        prediction = out_label.squeeze().detach().cpu().numpy()
        metrics = multilabel_metrics(prediction, gt)
        metrics_all.append(metrics)
        print(metrics)

        # zoom in refinement
        out_label_refined = None
        if network_crop is not None:
            rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
            if rgb_crop.shape[0] > 0:
                features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                labels_crop, selected_pixels_crop = clustering_features(features_crop)
                out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

        # evaluation
        if out_label_refined is not None:
            prediction_refined = out_label_refined.squeeze().detach().cpu().numpy()
        else:
            prediction_refined = prediction.copy()
        metrics_refined = multilabel_metrics(prediction_refined, gt)
        metrics_all_refined.append(metrics_refined)
        print(metrics_refined)

        if cfg.TEST.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, out_label_refined, features, 
                selected_pixels=selected_pixels, bbox=None)
        else:
            # save results
            result = {'labels': prediction, 'labels_refined': prediction_refined, 'filename': sample['filename']}
            filename = os.path.join(output_dir, '%06d.mat' % i)
            print(filename)
            scipy.io.savemat(filename, result, do_compression=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))

    # sum the values with same keys
    print('========================================================')
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================Refined=============================')

    result_refined = {}
    for metrics in metrics_all_refined:
        for k in metrics.keys():
            result_refined[k] = result_refined.get(k, 0) + metrics[k]

    for k in sorted(result_refined.keys()):
        result_refined[k] /= num
        print('%s: %f' % (k, result_refined[k]))
    print(result_refined)
    print('========================================================')
