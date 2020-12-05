# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import random
import numpy as np
import numbers
from PIL import Image # PyTorch likes PIL instead of cv2
import cv2

# My libraries
from utils import mask as util_
from fcn.config import cfg


##### Useful Utilities #####

def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

def translate(img, tx, ty, interpolation=cv2.INTER_LINEAR):
    """ Translate img by tx, ty

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    M = np.array([[1,0,tx],
                  [0,1,ty]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)

def rotate(img, angle, center=None, interpolation=cv2.INTER_LINEAR):
    """ Rotate img <angle> degrees counter clockwise w.r.t. center of image

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    if center is None:
        center = (W//2, H//2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)


##### Depth Augmentations #####

def add_noise_to_depth(depth_img, noise_params):
    """ Distort depth image with multiplicative gamma noise.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
    depth_img = multiplicative_noise * depth_img

    return depth_img

def add_noise_to_xyz(xyz_img, depth_img, noise_params):
    """ Add (approximate) Gaussian Process noise to ordered point cloud.
        This is adapted from the DexNet 2.0 codebase.

        @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    small_H, small_W = (np.array([H, W]) / noise_params['gp_rescale_factor']).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

    return xyz_img

def dropout_random_ellipses(depth_img, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        mask = np.zeros_like(depth_img)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        depth_img[mask == 1] = 0

    return depth_img


##### RGB Augmentations #####

def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes

        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

def random_color_warp(image, d_h=None, d_s=None, d_l=None):
    """ Given an RGB image [H x W x 3], add random hue, saturation and luminosity to the image

        Code adapted from: https://github.com/yuxng/PoseCNN/blob/master/lib/utils/blob.py
    """
    H, W, _ = image.shape

    image_color_warped = np.zeros_like(image)

    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (random.random() - 0.5) * 0.2 * 256
    if d_l is None:
        d_l = (random.random() - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (random.random() - 0.5) * 0.2 * 256

    # Convert the RGB to HLS
    hls = cv2.cvtColor(image.round().astype(np.uint8), cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)

    # Add the values to the image H, L, S
    new_h = (np.round((h + d_h)) % 256).astype(np.uint8)
    new_l = np.round(np.clip(l + d_l, 0, 255)).astype(np.uint8)
    new_s = np.round(np.clip(s + d_s, 0, 255)).astype(np.uint8)

    # Convert the HLS to RGB
    new_hls = cv2.merge((new_h, new_l, new_s)).astype(np.uint8)
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)

    image_color_warped = new_im.astype(np.float32)

    return image_color_warped

def random_horizontal_flip(image, label):
    """Randomly horizontally flip the image/label w.p. 0.5

        @param image: a [H x W x 3] numpy array
        @param label: a [H x W] numpy array
    """

    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()

    return image, label


##### Label transformations #####

def random_morphological_transform(label):
    """ Randomly erode/dilate the label

        @param label: a [H x W] numpy array of {0, 1}
    """

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= cfg.TRAIN.max_augmentation_tries:
            print('Morph: Exhausted number of augmentation tries...')
            return label

        # Sample whether we do erosion or dilation, and kernel size for that
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        sidelength = np.mean([x_max - x_min, y_max - y_min])

        morphology_kernel_size = 0; num_ksize_tries = 0;
        while morphology_kernel_size == 0:
            if num_ksize_tries >= 50: # 50 tries for this
                print('Morph: Exhausted number of augmentation tries... Sidelength: {sidelength}')
                return label

            dilation_percentage = np.random.beta(cfg.TRAIN.label_dilation_alpha, cfg.TRAIN.label_dilation_beta)
            morphology_kernel_size = int(round(sidelength * dilation_percentage))

            num_ksize_tries += 1

        iterations = np.random.randint(1, cfg.TRAIN.morphology_max_iters+1)

        # Erode/dilate the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
        if np.random.rand() < 0.5:
            morphed_label = cv2.erode(label, kernel, iterations=iterations)
        else:
            morphed_label = cv2.dilate(label, kernel, iterations=iterations)

        # Make sure there the mass is reasonable
        if (np.count_nonzero(morphed_label) / morphed_label.size > 0.001) and \
           (np.count_nonzero(morphed_label) / morphed_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return morphed_label

def random_ellipses(label):
    """ Randomly add/drop a few ellipses in the mask
        This is adapted from the DexNet 2.0 code.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param label: a [H x W] numpy array of {0, 1}
    """
    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= cfg.TRAIN.max_augmentation_tries:
            print('Ellipse: Exhausted number of augmentation tries...')
            return label

        new_label = label.copy()

        # Sample number of ellipses to include/dropout
        num_ellipses = np.random.poisson(cfg.TRAIN.num_ellipses_mean)

        # Sample ellipse centers by sampling from Gaussian at object center
        pixel_indices = util_.build_matrix_of_indices(H, W)
        h_idx, w_idx = np.where(new_label)
        mu = np.mean(pixel_indices[h_idx, w_idx, :], axis=0) # Shape: [2]. y_center, x_center
        sigma = 2*np.cov(pixel_indices[h_idx, w_idx, :].T) # Shape: [2 x 2]
        if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
            print(mu, sigma, h_idx, w_idx)
        ellipse_centers = np.random.multivariate_normal(mu, sigma, size=num_ellipses) # Shape: [num_ellipses x 2]
        ellipse_centers = np.round(ellipse_centers).astype(int)

        # Sample ellipse radii and angles
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(new_label)
        scale_factor = max(x_max - x_min, y_max - y_min) * cfg.TRAIN.ellipse_size_percentage # Mean of gamma r.v.
        x_radii = np.random.gamma(cfg.TRAIN.ellipse_gamma_base_shape * scale_factor, 
                                  cfg.TRAIN.ellipse_gamma_base_scale, 
                                  size=num_ellipses)
        y_radii = np.random.gamma(cfg.TRAIN.ellipse_gamma_base_shape * scale_factor, 
                                  cfg.TRAIN.ellipse_gamma_base_scale, 
                                  size=num_ellipses)
        angles = np.random.randint(0, 360, size=num_ellipses)

        # Dropout ellipses
        for i in range(num_ellipses):
            center = ellipse_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            # include or dropout the ellipse
            mask = np.zeros_like(new_label)
            mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
            if np.random.rand() < 0.5:
                new_label[mask == 1] = 0 # Drop out ellipse
            else:
                new_label[mask == 1] = 1 # Add ellipse

        # Make sure the mass is reasonable
        if (np.count_nonzero(new_label) / new_label.size > 0.001) and \
           (np.count_nonzero(new_label) / new_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return new_label

def random_translation(label):
    """ Randomly translate mask

        @param label: a [H x W] numpy array of {0, 1}
    """

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= cfg.TRAIN.max_augmentation_tries:
            print('Translate: Exhausted number of augmentation tries...')
            return label

        # Get tight bbox of mask
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        sidelength = max(x_max - x_min, y_max - y_min)

        # sample translation pixels
        translation_percentage = np.random.beta(cfg.TRAIN.translation_alpha, cfg.TRAIN.translation_beta)
        translation_percentage = max(translation_percentage, cfg.TRAIN.translation_percentage_min)
        translation_max = int(round(translation_percentage * sidelength))
        translation_max = max(translation_max, 1) # To make sure things don't error out

        tx = np.random.randint(-translation_max, translation_max)
        ty = np.random.randint(-translation_max, translation_max)   

        translated_label = translate(label, tx, ty, interpolation=cv2.INTER_NEAREST)

        # Make sure the mass is reasonable
        if (np.count_nonzero(translated_label) / translated_label.size > 0.001) and \
           (np.count_nonzero(translated_label) / translated_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return translated_label


def random_rotation(label):
    """ Randomly rotate mask

        @param label: a [H x W] numpy array of {0, 1}
    """
    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= cfg.TRAIN.max_augmentation_tries:
            print('Rotate: Exhausted number of augmentation tries...')
            return label

        # Rotate about center of box
        pixel_indices = util_.build_matrix_of_indices(H, W)
        h_idx, w_idx = np.where(label)
        mean = np.mean(pixel_indices[h_idx, w_idx, :], axis=0) # Shape: [2]. y_center, x_center

        # Sample an angle
        applied_angle = np.random.uniform(-cfg.TRAIN.rotation_angle_max, cfg.TRAIN.rotation_angle_max)

        rotated_label = rotate(label, applied_angle, center=tuple(mean[::-1]), interpolation=cv2.INTER_NEAREST)
        # Make sure the mass is reasonable
        if (np.count_nonzero(rotated_label) / rotated_label.size > 0.001) and \
           (np.count_nonzero(rotated_label) / rotated_label.size < 0.98):
            valid_transform = True
        num_tries += 1

    return rotated_label


def random_cut(label):
    """ Randomly cut part of mask

        @param label: a [H x W] numpy array of {0, 1}
    """

    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= cfg.TRAIN.max_augmentation_tries:
            print('Cut: Exhausted number of augmentation tries...')
            return label

        cut_label = label.copy()

        # Sample cut percentage
        cut_percentage = np.random.uniform(cfg.TRAIN.cut_percentage_min, cfg.TRAIN.cut_percentage_max)
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        if np.random.rand() < 0.5: # choose width
            
            sidelength = x_max - x_min
            if np.random.rand() < 0.5:  # from the left
                x = int(round(cut_percentage * sidelength)) + x_min
                cut_label[y_min:y_max+1, x_min:x] = 0
            else: # from the right
                x = x_max - int(round(cut_percentage * sidelength))
                cut_label[y_min:y_max+1, x:x_max+1] = 0

        else: # choose height
            
            sidelength = y_max - y_min
            if np.random.rand() < 0.5:  # from the top
                y = int(round(cut_percentage * sidelength)) + y_min
                cut_label[y_min:y, x_min:x_max+1] = 0
            else: # from the bottom
                y = y_max - int(round(cut_percentage * sidelength))
                cut_label[y:y_max+1, x_min:x_max+1] = 0

        # Make sure the mass is reasonable
        if (np.count_nonzero(cut_label) / cut_label.size > 0.001) and \
           (np.count_nonzero(cut_label) / cut_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return cut_label


def random_add(label):
    """ Randomly add part of mask 

        @param label: a [H x W] numpy array of {0, 1}
    """
    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:
        if num_tries >= cfg.TRAIN.max_augmentation_tries:
            print('Add: Exhausted number of augmentation tries...')
            return label

        added_label = label.copy()

        # Sample add percentage
        add_percentage = np.random.uniform(cfg.TRAIN.add_percentage_min, cfg.TRAIN.add_percentage_max)
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)

        # Sample translation from center
        translation_percentage_x = np.random.uniform(0, 2*add_percentage)
        tx = int(round( (x_max - x_min) * translation_percentage_x ))
        translation_percentage_y = np.random.uniform(0, 2*add_percentage)
        ty = int(round( (y_max - y_min) * translation_percentage_y ))

        if np.random.rand() < 0.5: # choose x direction

            sidelength = x_max - x_min
            ty = np.random.choice([-1, 1]) * ty # mask will be moved to the left/right. up/down doesn't matter

            if np.random.rand() < 0.5: # mask copied from the left. 
                x = int(round(add_percentage * sidelength)) + x_min
                try:
                    temp = added_label[y_min+ty : y_max+1+ty, x_min-tx : x-tx]
                    added_label[y_min+ty : y_max+1+ty, x_min-tx : x-tx] = np.logical_or(temp, added_label[y_min : y_max+1, x_min : x])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue
            else: # mask copied from the right
                x = x_max - int(round(add_percentage * sidelength))
                try:
                    temp = added_label[y_min+ty : y_max+1+ty, x+tx : x_max+1+tx]
                    added_label[y_min+ty : y_max+1+ty, x+tx : x_max+1+tx] = np.logical_or(temp, added_label[y_min : y_max+1, x : x_max+1])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue

        else: # choose y direction

            sidelength = y_max - y_min
            tx = np.random.choice([-1, 1]) * tx # mask will be moved up/down. lef/right doesn't matter

            if np.random.rand() < 0.5:  # from the top
                y = int(round(add_percentage * sidelength)) + y_min
                try:
                    temp = added_label[y_min-ty : y-ty, x_min+tx : x_max+1+tx]
                    added_label[y_min-ty : y-ty, x_min+tx : x_max+1+tx] = np.logical_or(temp, added_label[y_min : y, x_min : x_max+1])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue
            else: # from the bottom
                y = y_max - int(round(add_percentage * sidelength))
                try:
                    temp = added_label[y+ty : y_max+1+ty, x_min+tx : x_max+1+tx]
                    added_label[y+ty : y_max+1+ty, x_min+tx : x_max+1+tx] = np.logical_or(temp, added_label[y : y_max+1, x_min : x_max+1])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue

        # Make sure the mass is reasonable
        if (np.count_nonzero(added_label) / added_label.size > 0.001) and \
           (np.count_nonzero(added_label) / added_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return added_label
