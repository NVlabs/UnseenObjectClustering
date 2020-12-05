# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch

def log_softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        d = input - m.repeat(1, num_classes, 1, 1)
    else:
        d = input - m.repeat(1, num_classes)
    e = torch.exp(d)
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = d - torch.log(s.repeat(1, num_classes, 1, 1))
    else:
        output = d - torch.log(s.repeat(1, num_classes))
    return output


def softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        e = torch.exp(input - m.repeat(1, num_classes, 1, 1))
    else:
        e = torch.exp(input - m.repeat(1, num_classes))
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = torch.div(e, s.repeat(1, num_classes, 1, 1))
    else:
        output = torch.div(e, s.repeat(1, num_classes))
    return output


def concatenate_spatial_coordinates(feature_map):
    """ Adds x,y coordinates as channels to feature map

        @param feature_map: a [T x C x H x W] torch tensor
    """
    T, C, H, W = feature_map.shape

    # build matrix of indices. then replicated it T times
    MoI = build_matrix_of_indices(H, W) # Shape: [H, W, 2]
    MoI = np.tile(MoI, (T, 1, 1, 1)) # Shape: [T, H, W, 2]
    MoI[..., 0] = MoI[..., 0] / (H-1) * 2 - 1 # in [-1, 1]
    MoI[..., 1] = MoI[..., 1] / (W-1) * 2 - 1
    MoI = torch.from_numpy(MoI).permute(0,3,1,2).to(feature_map.device) # Shape: [T, 2, H, W]

    # Concatenate on the channels dimension
    feature_map = torch.cat([feature_map, MoI], dim=1)

    return feature_map
