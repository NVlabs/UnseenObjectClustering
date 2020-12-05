# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

########## Embedding Loss ##########

def zero_diagonal(x):
    """ Sets diagonal elements of x to 0

        @param x: a [batch_size x S x S] torch.FloatTensor
    """
    S = x.shape[1]
    return x * (1- torch.eye(S).to(x.device))


def compute_cluster_mean(x, cluster_masks, K, normalize):
    """ Computes the spherical mean of a set of unit vectors. This is a PyTorch implementation
        The definition of spherical mean is minimizes cosine similarity 
            to a set of points instead of squared error.

        Solves this problem:

            argmax_{||w||^2 <= 1} (sum_i x_i)^T w

        Turns out the solution is: S_n / ||S_n||, where S_n = sum_i x_i. 
            If S_n = 0, w can be anything.


        @param x: a [batch_size x C x H x W] torch.FloatTensor of N NORMALIZED C-dimensional unit vectors
        @param cluster_masks: a [batch_size x K x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}.
                              Note: cluster -1 (i.e. no cluster assignment) is ignored
        @param K: number of clusters

        @return: a [batch_size x C x K] torch.FloatTensor of NORMALIZED cluster means
    """
    batch_size, C = x.shape[:2]
    cluster_means = torch.zeros((batch_size, C, K), device=x.device)
    for k in range(K):
        mask = (cluster_masks == k).float() # Shape: [batch_size x 1 x H x W]
        # adding 1e-10 because if mask has nothing, it'll hit NaNs
        # * here is broadcasting
        cluster_means[:,:,k] = torch.sum(x * mask, dim=[2, 3]) / (torch.sum(mask, dim=[2, 3]) + 1e-10) 

    # normalize to compute spherical mean
    if normalize:
        cluster_means = F.normalize(cluster_means, p=2, dim=1) # Note, if any vector is zeros, F.normalize will return the zero vector
    return cluster_means


class EmbeddingLoss(nn.Module):

    def __init__(self, alpha, delta, lambda_intra, lambda_inter, metric='cosine', normalize=True):
        super(EmbeddingLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.metric = metric
        self.normalize = normalize

    def forward(self, x, cluster_masks):
        """ Compute the clustering loss. Assumes the batch is a sequence of consecutive frames

            @param x: a [batch_size x C x H x W] torch.FloatTensor of pixel embeddings
            @param cluster_masks: a [batch_size x 1 x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}
        """

        batch_size = x.shape[0]
        K = int(cluster_masks.max().item()) + 1

        # Compute cluster means across batch dimension
        cluster_means = compute_cluster_mean(x, cluster_masks, K, self.normalize) # Shape: [batch_size x C x K]

        ### Intra cluster loss ###

        # Tile the cluster means appropriately. Also calculate number of pixels per mask for pixel weighting
        tiled_cluster_means = torch.zeros_like(x, device=x.device) # Shape: [batch_size x C x H x W]
        for k in range(K):
            mask = (cluster_masks == k).float() # Shape: [batch_size x 1 x H x W]
            tiled_cluster_means += mask * cluster_means[:,:,k].unsqueeze(2).unsqueeze(3)

        # ignore label -1
        labeled_embeddings = (cluster_masks >= 0).squeeze(1).float() # Shape: [batch_size x H x W]

        # Compute distance to cluster center
        if self.metric == 'cosine':
            intra_cluster_distances = labeled_embeddings * (0.5 * (1 - torch.sum(x * tiled_cluster_means, dim=1))) # Shape: [batch_size x H x W]
        elif self.metric == 'euclidean':
            intra_cluster_distances = labeled_embeddings * (torch.norm(x - tiled_cluster_means, dim=1))

        # Hard Negative Mining
        intra_cluster_mask = (intra_cluster_distances - self.alpha) > 0
        intra_cluster_mask = intra_cluster_mask.float()
        if torch.sum(intra_cluster_mask) > 0:
            intra_cluster_loss = torch.pow(intra_cluster_distances, 2)

            # calculate datapoint_weights
            datapoint_weights = torch.zeros((batch_size,) + intra_cluster_distances.shape[1:], device=x.device)
            for k in range(K):
                # find number of datapoints in cluster k that are > alpha away from cluster center
                mask = (cluster_masks == k).float().squeeze(1) # Shape: [batch_size x H x W]
                N_k = torch.sum((intra_cluster_distances > self.alpha).float() * mask, dim=[1, 2], keepdim=True) # Shape: [batch_size x 1 x 1]
                datapoint_weights += mask * N_k
            datapoint_weights = torch.max(datapoint_weights, torch.FloatTensor([50]).to(x.device)) # Max it with 50 so it doesn't get too small
            datapoint_weights *= K

            intra_cluster_loss = torch.sum(intra_cluster_loss / datapoint_weights) / batch_size
        else:
            intra_cluster_loss = torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))
        intra_cluster_loss = self.lambda_intra * intra_cluster_loss

        ### Inter cluster loss ###
        if K > 1:
            if self.metric == 'cosine':
                # Shape: [batch_size x K x K]
                inter_cluster_distances = .5 * (1 - torch.sum(cluster_means.unsqueeze(2) * cluster_means.unsqueeze(3), dim=1))
            elif self.metric == 'euclidean':
                inter_cluster_distances = torch.norm(cluster_means.unsqueeze(2) - cluster_means.unsqueeze(3), dim=1)

            inter_cluster_loss = torch.sum(torch.pow(torch.clamp(zero_diagonal(self.delta - inter_cluster_distances), min=0), 2)) / (K*(K-1)/2 * batch_size)
            inter_cluster_loss = self.lambda_inter * inter_cluster_loss
        else:
            inter_cluster_loss = torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))

        loss = intra_cluster_loss + inter_cluster_loss
        return loss, intra_cluster_loss, inter_cluster_loss
