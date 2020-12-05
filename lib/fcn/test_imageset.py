# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn.functional as F
import time
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import normalize_descriptor
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.mean_shift import mean_shift_smart_init


def test_image_segmentation(ind, network, dataset, img, segmentor):
    """test on a single image"""

    height = img.shape[0]
    width = img.shape[1]

    # compute image blob
    inputs = img.astype(np.float32, copy=True)
    inputs -= cfg.PIXEL_MEANS
    inputs = np.transpose(inputs / 255.0, (2, 0, 1))
    inputs = inputs[np.newaxis, :, :, :]
    inputs = torch.from_numpy(inputs).cuda()

    # use fake label blob
    label = torch.cuda.FloatTensor(1, 2, height, width)

    # run network
    if network.module.embedding:
        features = network(inputs, label)
        out_label = torch.zeros((features.shape[0], height, width))

        # mean shift clustering
        num_seeds = 20
        kappa = 20
        for i in range(features.shape[0]):
            X = features[i].view(features.shape[1], -1)
            X = torch.transpose(X, 0, 1)
            cluster_labels, selected_indices = mean_shift_smart_init(X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric='cosine')
            out_label[i] = cluster_labels.view(height, width)
    else:
        out_label = network(inputs, label)

    # mask refinement
    if segmentor is not None:
        out_label_refined, out_label_crop, rgb_crop, roi = segmentor.refine(inputs, out_label.clone())
    else:
        out_label_refined = None
        roi = None

    if cfg.TEST.VISUALIZE:
        fig = plt.figure()
        m = 2
        n = 3
        start = 1
        if network.module.embedding:
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(img[:, :, (2, 1, 0)])
            for i in range(num_seeds):
                index = selected_indices[i]
                y = index / width
                x = index % width
                plt.plot(x, y, 'ro')
            ax.set_title('input')

            im = torch.cuda.FloatTensor(height, width, 3)
            for i in range(3):
                im[:, :, i] = torch.sum(features[0, i::3, :, :], dim=0)
            im = normalize_descriptor(im.detach().cpu().numpy())
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('features')

            ax = fig.add_subplot(m, n, start)
            start += 1
            out_label_blob = out_label.cpu().numpy()
            label = out_label_blob[0, :, :]
            plt.imshow(label)
            ax.set_title('cluster labels')

            if roi is not None:
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(img[:, :, (2, 1, 0)])
                for i in range(roi.shape[0]):
                    x1 = roi[i, 0]
                    y1 = roi[i, 1]
                    x2 = roi[i, 2]
                    y2 = roi[i, 3]
                    plt.gca().add_patch(
                        plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

            if segmentor is not None:
                ax = fig.add_subplot(m, n, start)
                start += 1
                out_label_blob = out_label_refined.cpu().numpy()
                label = out_label_blob[0, :, :]
                plt.imshow(label)
                ax.set_title('cluster labels refined')

            # mng = plt.get_current_fig_manager()
            # filename = 'output/images/%06d.png' % ind
            # fig.savefig(filename)
            plt.show()
        else:
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(img[:, :, (2, 1, 0)])

            # show out label
            out_label_blob = out_label.cpu().numpy()
            label = out_label_blob[0, :, :]
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(label)
            ax.set_title('out label')

        plt.show()

    return out_label, out_label_refined, roi
