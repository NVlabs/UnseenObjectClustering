# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import utils as util_

class Conv2d_GN_ReLU(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLU, self).__init__()
        padding = 0 if ksize < 2 else ksize//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=ksize, stride=stride, 
                               padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        return out

class Conv2d_GN_ReLUx2(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 
            conv2d + groupnorm + ReLU
            (and a possible downsampling operation)

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLUx2, self).__init__()
        self.layer1 = Conv2d_GN_ReLU(in_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)
        self.layer2 = Conv2d_GN_ReLU(out_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out

class Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(nn.Module):
    """ Implements a module that performs
            Upsample (reduction: conv2d + groupnorm + ReLU + bilinear_sampling) +
            concat + conv2d + groupnorm + ReLU 
        for the U-Net decoding architecture with an arbitrary number of encoders

        The Upsample operation consists of a Conv2d_GN_ReLU that reduces the channels by 2,
            followed by bilinear sampling

        Note: in_channels is number of channels of ONE of the inputs to the concatenation

    """
    def __init__(self, in_channels, out_channels, num_groups, num_encoders, ksize=3, stride=1):
        super(Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch, self).__init__()
        self.channel_reduction_layer = Conv2d_GN_ReLU(in_channels, in_channels//2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = Conv2d_GN_ReLU(int(in_channels//2 * (num_encoders+1)), out_channels, num_groups)

    def forward(self, x, skips):
        """ Forward module

            @param skips: a list of intermediate skip-layer torch tensors from each encoder
        """
        x = self.channel_reduction_layer(x)
        x = self.upsample(x)
        out = torch.cat([x] + skips, dim=1) # Concat on channels dimension
        out = self.conv_gn_relu(out)

        return out

def maxpool2x2(input, ksize=2, stride=2):
    """2x2 max pooling"""
    return nn.MaxPool2d(ksize, stride=stride)(input)


# another way to build encoder/decoder
def make_encoder_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    output_scale = 1.0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            output_scale /= 2.0
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2,inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.2,inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), in_channels, output_scale


def make_decoder_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for i in range(len(cfg)):
        v = cfg[i]
        if type(v) is str:
            if v[0] == 'd':
                v = int(v[1:])
                convtrans2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1)
                if batch_norm:
                    layers += [convtrans2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                else:
                    layers += [convtrans2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                in_channels = v
            elif v[0] == 'c':
                v = int(v[1:])
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
            elif v[0] == 'D':
                layers += [nn.Dropout(p=0.2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            else:
                # no relu for the last layer for embedding
                if i == len(cfg) - 1:
                    layers += [conv2d]
                else:
                    layers += [conv2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


################## Network Definitions ##################

class UNet_Encoder(nn.Module):
    
    def __init__(self, input_channels, feature_dim):
        super(UNet_Encoder, self).__init__()
        self.ic = input_channels
        self.fd = feature_dim
        self.build_network()
        
    def build_network(self):
        """ Build encoder network
            Uses a U-Net-like architecture
        """

        ### Encoder ###
        self.layer1 = Conv2d_GN_ReLUx2(self.ic, self.fd, self.fd)
        self.layer2 = Conv2d_GN_ReLUx2(self.fd, self.fd*2, self.fd)
        self.layer3 = Conv2d_GN_ReLUx2(self.fd*2, self.fd*4, self.fd)
        self.layer4 = Conv2d_GN_ReLUx2(self.fd*4, self.fd*8, self.fd)
        self.last_layer = Conv2d_GN_ReLU(self.fd*8, self.fd*16, self.fd)


    def forward(self, images):

        x1 = self.layer1(images)
        mp_x1 = maxpool2x2(x1)
        x2 = self.layer2(mp_x1)
        mp_x2 = maxpool2x2(x2)
        x3 = self.layer3(mp_x2)
        mp_x3 = maxpool2x2(x3)
        x4 = self.layer4(mp_x3)
        mp_x4 = maxpool2x2(x4)
        x5 = self.last_layer(mp_x4)

        return x5, [x1, x2, x3, x4]

class UNet_Decoder(nn.Module):
    """ A U-Net decoder that allows for multiple encoders
    """

    def __init__(self, num_encoders, feature_dim, coordconv=False):
        super(UNet_Decoder, self).__init__()
        self.ne = num_encoders
        self.fd = feature_dim
        self.coordconv = coordconv
        self.build_network()

    def build_network(self):
        """ Build a decoder network
            Uses a U-Net-like architecture
        """

        # Fusion layer
        self.fuse_layer = Conv2d_GN_ReLU(self.fd*16 * self.ne, self.fd*16, self.fd, ksize=1)

        # Decoding
        self.layer1 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*16, self.fd*8, self.fd, self.ne)
        self.layer2 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*8, self.fd*4, self.fd, self.ne)
        self.layer3 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*4, self.fd*2, self.fd, self.ne)
        self.layer4 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*2, self.fd, self.fd, self.ne)

        # Final layer
        self.layer5 = Conv2d_GN_ReLU(self.fd, self.fd, self.fd)

        if self.coordconv:
            # Extra 1x1 Conv layers for CoordConv
            self.layer6 = Conv2d_GN_ReLUx2(self.fd+2, self.fd, self.fd, ksize=1)
            self.layer7 = Conv2d_GN_ReLUx2(self.fd, self.fd, self.fd, ksize=1)        

        # This puts features everywhere, not just nonnegative orthant
        self.last_conv = nn.Conv2d(self.fd, self.fd, kernel_size=3,
                                   stride=1, padding=1, bias=True)

    def forward(self, encoder_list):
        """ Forward module

            @param encoder_list: a list of tuples
                                 each tuple includes 2 elements:
                                    - output of encoder: an [N x C x H x W] torch tensor
                                    - list of intermediate outputs: a list of 4 torch tensors

        """

        # Apply fusion layer to the concatenation of encoder outputs
        out = torch.cat([x[0] for x in encoder_list], dim=1) # Concatenate on channels dimension
        out = self.fuse_layer(out)

        out = self.layer1(out, [x[1][3] for x in encoder_list])
        out = self.layer2(out, [x[1][2] for x in encoder_list])
        out = self.layer3(out, [x[1][1] for x in encoder_list])
        out = self.layer4(out, [x[1][0] for x in encoder_list])

        out = self.layer5(out)

        if self.coordconv:
            out = util_.concatenate_spatial_coordinates(out)
            out = self.layer6(out)
            out = self.layer7(out)

        out = self.last_conv(out)

        return out
