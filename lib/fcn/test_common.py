# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import time
import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fcn.config import cfg
from utils.mask import visualize_segmentation


def normalize_descriptor(res, stats=None):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """

    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    normed_res = np.clip(res, res_min, res_max)
    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (normed_res - res_min) / scale
    return normed_res


def _vis_features(features, labels, rgb, intial_labels, selected_pixels=None):
    num = features.shape[0]
    height = features.shape[2]
    width = features.shape[3]
    fig = plt.figure()
    start = 1
    m = np.ceil((num * 4 ) / 8.0)
    n = 8
    im_blob = rgb.cpu().numpy()
    for i in range(num):
        if i < m * n / 4:

            # show image
            im = im_blob[i, :3, :, :].copy()
            im = im.transpose((1, 2, 0)) * 255.0
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = np.clip(im, 0, 255)
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('image')
            plt.axis('off')

            '''
            if selected_pixels is not None:
                selected_indices = selected_pixels[i]
                for j in range(len(selected_indices)):
                    index = selected_indices[j]
                    y = index / width
                    x = index % width
                    plt.plot(x, y, 'ro', markersize=1.0)
            '''

            im = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im[:, :, j] = torch.sum(features[i, j::3, :, :], dim=0)
            im = normalize_descriptor(im.detach().cpu().numpy())
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('features')
            plt.axis('off')

            ax = fig.add_subplot(m, n, start)
            start += 1
            label = labels[i].detach().cpu().numpy()
            plt.imshow(label)
            ax.set_title('labels')
            plt.axis('off')

            ax = fig.add_subplot(m, n, start)
            start += 1
            label = intial_labels[i].detach().cpu().numpy()
            plt.imshow(label)
            ax.set_title('intial labels')
            plt.axis('off')

    plt.show()


def _vis_minibatch_segmentation_final(image, depth, label, out_label=None, out_label_refined=None,
    features=None, ind=None, selected_pixels=None, bbox=None):

    if depth is None:
        im_blob = image.cpu().numpy()
    else:
        im_blob = image.cpu().numpy()
        depth_blob = depth.cpu().numpy()

    num = im_blob.shape[0]
    height = im_blob.shape[2]
    width = im_blob.shape[3]

    if label is not None:
        label_blob = label.cpu().numpy()
    if out_label is not None:
        out_label_blob = out_label.cpu().numpy()
    if out_label_refined is not None:
        out_label_refined_blob = out_label_refined.cpu().numpy()

    m = 2
    n = 3
    for i in range(num):

        # image
        im = im_blob[i, :3, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        fig = plt.figure()
        start = 1
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image')
        plt.axis('off')

        # depth
        if depth is not None:
            depth = depth_blob[i][2]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(depth)
            ax.set_title('depth')
            plt.axis('off')

        # feature
        if features is not None:
            im_feature = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im_feature[:, :, j] = torch.sum(features[i, j::3, :, :], dim=0)
            im_feature = normalize_descriptor(im_feature.detach().cpu().numpy())
            im_feature *= 255
            im_feature = im_feature.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_feature)
            ax.set_title('feature map')
            plt.axis('off')

        # initial seeds
        if selected_pixels is not None:
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('initial seeds')
            plt.axis('off')
            selected_indices = selected_pixels[i]
            for j in range(len(selected_indices)):
                index = selected_indices[j]
                y = index / width
                x = index % width
                plt.plot(x, y, 'ro', markersize=2.0)

        # intial mask
        mask = out_label_blob[i, :, :]
        im_label = visualize_segmentation(im, mask, return_rgb=True)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('initial label')
        plt.axis('off')

        # refined mask
        if out_label_refined is not None:
            mask = out_label_refined_blob[i, :, :]
            im_label = visualize_segmentation(im, mask, return_rgb=True)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_label)
            ax.set_title('refined label')
            plt.axis('off')
        elif label is not None:
            # show gt label
            mask = label_blob[i, 0, :, :]
            im_label = visualize_segmentation(im, mask, return_rgb=True)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_label)
            ax.set_title('gt label')
            plt.axis('off')

        if ind is not None:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.pause(0.001)
            # plt.show(block=False)
            filename = 'output/images/%06d.png' % ind
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()


def _vis_minibatch_segmentation(image, depth, label, out_label=None, out_label_refined=None,
    features=None, ind=None, selected_pixels=None, bbox=None):

    if depth is None:
        im_blob = image.cpu().numpy()
        m = 2
        n = 3
    else:
        im_blob = image.cpu().numpy()
        depth_blob = depth.cpu().numpy()
        m = 3
        n = 3

    num = im_blob.shape[0]
    height = im_blob.shape[2]
    width = im_blob.shape[3]
    if label is not None:
        label_blob = label.cpu().numpy()
    if out_label is not None:
        out_label_blob = out_label.cpu().numpy()
    if out_label_refined is not None:
        out_label_refined_blob = out_label_refined.cpu().numpy()

    for i in range(num):

        # image
        im = im_blob[i, :3, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)

        '''
        if out_label_refined is not None:
            mask = out_label_refined_blob[i, :, :]
            visualize_segmentation(im, mask)
        #'''

        # show image
        fig = plt.figure()
        start = 1
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image')
        plt.axis('off')

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        plt.axis('off')

        if bbox is not None:
            boxes = bbox[i].numpy()
            for j in range(boxes.shape[0]):
                x1 = boxes[j, 0]
                y1 = boxes[j, 1]
                x2 = boxes[j, 2]
                y2 = boxes[j, 3]
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        if selected_pixels is not None:
            selected_indices = selected_pixels[i]
            for j in range(len(selected_indices)):
                index = selected_indices[j]
                y = index / width
                x = index % width
                plt.plot(x, y, 'ro', markersize=1.0)

        if im_blob.shape[1] == 4:
            label = im_blob[i, 3, :, :]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('initial label')

        if depth is not None:
            depth = depth_blob[i]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(depth[0])
            ax.set_title('depth X')
            plt.axis('off')
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(depth[1])
            ax.set_title('depth Y')
            plt.axis('off')
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(depth[2])
            ax.set_title('depth Z')
            plt.axis('off')

        # show label
        if label is not None:
            label = label_blob[i, 0, :, :]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('gt label')
            plt.axis('off')

        # show out label
        if out_label is not None:
            label = out_label_blob[i, :, :]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('out label')
            plt.axis('off')

        # show out label refined
        if out_label_refined is not None:
            label = out_label_refined_blob[i, :, :]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('out label refined')
            plt.axis('off')

        if features is not None:
            im = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im[:, :, j] = torch.sum(features[i, j::3, :, :], dim=0)
            im = normalize_descriptor(im.detach().cpu().numpy())
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('features')
            plt.axis('off')

        if ind is not None:
            mng = plt.get_current_fig_manager()
            plt.show()
            filename = 'output/images/%06d.png' % ind
            fig.savefig(filename)

        plt.show()
