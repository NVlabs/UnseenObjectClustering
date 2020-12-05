# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys, os
from itertools import compress
import torch

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import cv2
from PIL import Image


def get_color_mask(object_index, nc=None):
    """ Colors each index differently. Useful for visualizing semantic masks

        @param object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
        @param nc: total number of colors. If None, this will be inferred by masks
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
        
    return color_mask

def build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.

        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)


def visualize_segmentation(im, masks, nc=None, return_rgb=False, save_dir=None):
    """ Visualize segmentations nicely. Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

        @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
        @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
        @param nc: total number of colors. If None, this will be inferred by masks
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    if not return_rgb:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    if not return_rgb:
        # matplotlib stuff
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        try:
            contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        except:
            im2, contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(c.reshape((-1, 2)), fill=False, facecolor=color_mask, edgecolor='w', linewidth=1.2, alpha=0.5)
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255,255,255), 2)


    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image
    

### These two functions were adatped from the DAVIS public dataset ###

def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation

def imwrite_indexed(filename,array):
    """ Save indexed png with palette."""

    palette_abspath = '/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def mask_to_tight_box_numpy(mask):
    """ Return bbox given mask

        @param mask: a [H x W] numpy array
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max

def mask_to_tight_box_pytorch(mask):
    """ Return bbox given mask

        @param mask: a [H x W] torch tensor
    """
    a = torch.nonzero(mask)
    bbox = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max

def mask_to_tight_box(mask):
    if type(mask) == torch.Tensor:
        return mask_to_tight_box_pytorch(mask)
    elif type(mask) == np.ndarray:
        return mask_to_tight_box_numpy(mask)
    else:
        raise Exception("Data type {type(mask)} not understood for mask_to_tight_box...")
