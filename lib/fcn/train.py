# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *

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


def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """
    cross_entropy = -torch.sum(labels * scores, dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)
    return loss


def train_segnet(train_loader, background_loader, network, optimizer, epoch, embedding=False, rrn=False):

    batch_time = AverageMeter()
    epoch_size = len(train_loader)
    enum_background = enumerate(background_loader)

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        # construct input
        image = sample['image_color'].cuda()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth = sample['depth'].cuda()
        else:
            depth = None

        label = sample['label'].cuda()
        if rrn:
            initial_mask = sample['initial_mask'].cuda()

        if cfg.TRAIN.CHANGE_BACKGROUND:
            mask = sample['mask'].cuda()
            # add background
            try:
                _, background = next(enum_background)
            except:
                enum_background = enumerate(background_loader)
                _, background = next(enum_background)

            num = image.size(0)
            if background['background_color'].size(0) < num:
                enum_background = enumerate(background_loader)
                _, background = next(enum_background)

            if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
                background_color = background['background_color'].cuda()
                image = mask * image + (1 - mask) * background_color[:num]

        if embedding:
            loss, intra_cluster_loss, inter_cluster_loss, features = network(image, label, depth)
            loss = torch.sum(loss)
            intra_cluster_loss = torch.sum(intra_cluster_loss)
            inter_cluster_loss = torch.sum(inter_cluster_loss)
            out_label = None
        else:
            if rrn:
                image = torch.cat([image, initial_mask], dim=1) # Shape: [N x 4 x H x W]

            out_logsoftmax, out_weight, out_label = network(image, label)
            loss = loss_cross_entropy(out_logsoftmax, out_weight)
            features = None

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, features=features)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if embedding:
            print('[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, lr %.6f, time %.2f' \
                % (epoch, cfg.epochs, i, epoch_size, loss, intra_cluster_loss, inter_cluster_loss, optimizer.param_groups[0]['lr'], batch_time.val))
        else:
            print('[%d/%d][%d/%d], loss %.4f, lr %.6f, time %.2f' \
                % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1
