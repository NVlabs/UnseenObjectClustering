# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys
import torch
import torch.nn.functional as F
import numpy as np
from fcn.config import cfg

def ball_kernel(Z, X, kappa, metric='cosine'):
    """ Computes pairwise ball kernel (without normalizing constant)
        (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints - the seeds
        @param X: a [m x d] torch.FloatTensor of NORMALIZED datapoints - the points

        @return: a [n x m] torch.FloatTensor of pairwise ball kernel computations,
                 without normalizing constant
    """
    if metric == 'euclidean':
        distance = Z.unsqueeze(1) - X.unsqueeze(0)
        distance = torch.norm(distance, dim=2)
        kernel = torch.exp(-kappa * torch.pow(distance, 2))
    elif metric == 'cosine':
        kernel = torch.exp(kappa * torch.mm(Z, X.t()))
    return kernel


def get_label_mode(array):
    """ Computes the mode of elements in an array.
        Ties don't matter. Ties are broken by the smallest value (np.argmax defaults)

        @param array: a numpy array
    """
    labels, counts = np.unique(array, return_counts=True)
    mode = labels[np.argmax(counts)].item()
    return mode


def connected_components(Z, epsilon, metric='cosine'):
    """
        For the connected components, we simply perform a nearest neighbor search in order:
            for each point, find the points that are up to epsilon away (in cosine distance)
            these points are labeled in the same cluster.

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints

        @return: a [n] torch.LongTensor of cluster labels
    """
    n, d = Z.shape

    K = 0
    cluster_labels = torch.ones(n, dtype=torch.long) * -1
    for i in range(n):
        if cluster_labels[i] == -1:

            if metric == 'euclidean':
                distances = Z.unsqueeze(1) - Z[i:i + 1].unsqueeze(0)  # a are points, b are seeds
                distances = torch.norm(distances, dim=2)
            elif metric == 'cosine':
                distances = 0.5 * (1 - torch.mm(Z, Z[i:i+1].t()))
            component_seeds = distances[:, 0] <= epsilon

            # If at least one component already has a label, then use the mode of the label
            if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                temp = cluster_labels[component_seeds].numpy()
                temp = temp[temp != -1]
                label = torch.tensor(get_label_mode(temp))
            else:
                label = torch.tensor(K)
                K += 1  # Increment number of clusters

            cluster_labels[component_seeds] = label

    return cluster_labels


def seed_hill_climbing_ball(X, Z, kappa, max_iters=10, metric='cosine'):
    """ Runs mean shift hill climbing algorithm on the seeds.
        The seeds climb the distribution given by the KDE of X

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the ball kernel
    """
    n, d = X.shape
    m = Z.shape[0]

    for _iter in range(max_iters):

        # Create a new object for Z
        new_Z = Z.clone()

        W = ball_kernel(Z, X, kappa, metric=metric)

        # use this allocated weight to compute the new center
        new_Z = torch.mm(W, X)  # Shape: [n x d]

        # Normalize the update
        if metric == 'euclidean':
            summed_weights = W.sum(dim=1)
            summed_weights = summed_weights.unsqueeze(1)
            summed_weights = torch.clamp(summed_weights, min=1.0)
            Z = new_Z / summed_weights
        elif metric == 'cosine':
            Z = F.normalize(new_Z, p=2, dim=1)

    return Z


def mean_shift_with_seeds(X, Z, kappa, max_iters=10, metric='cosine'):
    """ Runs mean-shift

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the von Mises-Fisher distribution
    """

    Z = seed_hill_climbing_ball(X, Z, kappa, max_iters=max_iters, metric=metric)

    # Connected components
    cluster_labels = connected_components(Z, 2 * cfg.TRAIN.EMBEDDING_ALPHA, metric=metric)  # Set epsilon = 0.1 = 2*alpha

    return cluster_labels, Z


def select_smart_seeds(X, num_seeds, return_selected_indices=False, init_seeds=None, num_init_seeds=None, metric='cosine'):
    """ Selects seeds that are as far away as possible

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param num_seeds: number of seeds to pick
        @param init_seeds: a [num_seeds x d] vector of initial seeds
        @param num_init_seeds: the number of seeds already chosen.
                               the first num_init_seeds rows of init_seeds have been chosen already

        @return: a [num_seeds x d] matrix of seeds
                 a [n x num_seeds] matrix of distances
    """
    n, d = X.shape
    selected_indices = -1 * torch.ones(num_seeds, dtype=torch.long)

    # Initialize seeds matrix
    if init_seeds is None:
        seeds = torch.empty((num_seeds, d), device=X.device)
        num_chosen_seeds = 0
    else:
        seeds = init_seeds
        num_chosen_seeds = num_init_seeds

    # Keep track of distances
    distances = torch.empty((n, num_seeds), device=X.device)

    if num_chosen_seeds == 0:  # Select first seed if need to
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed
        if metric == 'euclidean':
            distances[:, 0] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[:, 0] = 0.5 * (1 - torch.mm(X, selected_seed.unsqueeze(1))[:,0])  
        num_chosen_seeds += 1
    else:  # Calculate distance to each already chosen seed
        for i in range(num_chosen_seeds):
            if metric == 'euclidean':
                distances[:, i] = torch.norm(X - seeds[i:i+1, :], dim=1)
            elif metric == 'cosine':
                distances[:, i] = 0.5 * (1 - torch.mm(X, seeds[i:i+1, :].t())[:, 0])

    # Select rest of seeds
    for i in range(num_chosen_seeds, num_seeds):
        # Find the point that has the furthest distance from the nearest seed
        distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: [n]
        selected_seed_index = torch.argmax(distance_to_nearest_seed)
        selected_indices[i] = selected_seed_index
        selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
        seeds[i, :] = selected_seed

        # Calculate distance to this selected seed
        if metric == 'euclidean':
            distances[:, i] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[:, i] = 0.5 * (1 - torch.mm(X, selected_seed.unsqueeze(1))[:,0])

    return_tuple = (seeds,)
    if return_selected_indices:
        return_tuple += (selected_indices,)
    return return_tuple


def mean_shift_smart_init(X, kappa, num_seeds=100, max_iters=10, metric='cosine'):
    """ Runs mean shift with carefully selected seeds

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param dist_threshold: parameter for the von Mises-Fisher distribution
        @param num_seeds: number of seeds used for mean shift clustering

        @return: a [n] array of cluster labels
    """

    n, d = X.shape
    seeds, selected_indices = select_smart_seeds(X, num_seeds, return_selected_indices=True, metric=metric)
    seed_cluster_labels, updated_seeds = mean_shift_with_seeds(X, seeds, kappa, max_iters=max_iters, metric=metric)

    # Get distances to updated seeds
    if metric == 'euclidean':
        distances = X.unsqueeze(1) - updated_seeds.unsqueeze(0)  # a are points, b are seeds
        distances = torch.norm(distances, dim=2)
    elif metric == 'cosine':
        distances = 0.5 * (1 - torch.mm(X, updated_seeds.t())) # Shape: [n x num_seeds]

    # Get clusters by assigning point to closest seed
    closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
    cluster_labels = seed_cluster_labels[closest_seed_indices]

    # assign zero to the largest cluster
    num = len(torch.unique(seed_cluster_labels))
    count = torch.zeros(num, dtype=torch.long)
    for i in range(num):
        count[i] = (cluster_labels == i).sum()
    label_max = torch.argmax(count)
    if label_max != 0:
        index1 = cluster_labels == 0
        index2 = cluster_labels == label_max
        cluster_labels[index1] = label_max
        cluster_labels[index2] = 0

    return cluster_labels, selected_indices
