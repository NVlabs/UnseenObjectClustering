# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from fcn.config import cfg
from networks.utils import log_softmax_high_dimension, softmax_high_dimension
from networks.embedding import EmbeddingLoss
from . import unets
from . import resnet_dilated

__all__ = [
    'seg_vgg_embedding', 'seg_unet_embedding', 'seg_resnet34_8s_embedding_early',
    'seg_resnet34_8s_embedding', 'seg_resnet50_8s_embedding',
]

encoder_archs = {
    'vgg16-based-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 1024, 1024]
}

# Segmentation Network
class SEGNET(nn.Module):
    '''SEGNET a Encoder-Decoder for Object Segmentation.'''
    def __init__(self, init_weights=True, batch_norm=False, in_channels=3,
                 network_name='vgg', num_units=64, use_coordconv=False):
        super(SEGNET, self).__init__()

        self.network_name = network_name
        self.in_channels = in_channels
        self.metric = cfg.TRAIN.EMBEDDING_METRIC
        self.normalize = cfg.TRAIN.EMBEDDING_NORMALIZATION
        self.input_type = cfg.INPUT
        self.fusion_type = cfg.TRAIN.FUSION_TYPE
        self.embedding_pretrain = cfg.TRAIN.EMBEDDING_PRETRAIN

        # embedding loss
        alpha = cfg.TRAIN.EMBEDDING_ALPHA
        delta = cfg.TRAIN.EMBEDDING_DELTA
        lambda_intra = cfg.TRAIN.EMBEDDING_LAMBDA_INTRA
        lambda_inter = cfg.TRAIN.EMBEDDING_LAMBDA_INTER
        self.embedding_loss = EmbeddingLoss(alpha, delta, lambda_intra, lambda_inter, self.metric, self.normalize)

        decoder_archs = {
            'd16': [1024, 'd512', 512, 512, 'D', 'd512', 512, 512, 'D', 'd256', 256, 256, 'd128', 128, 128, 'd64', 64, 64, 'c2'],
            'd16-embedding': [1024, 'd512', 512, 512, 'D', 'd512', 512, 512, 'D', 'd256', 256, 256, 'd128', 128, 128, 'd64', 64, num_units],
        }

        if network_name == 'vgg':
            # encoder
            en_layers, en_out_channels, en_output_scale = unets.make_encoder_layers(encoder_archs['vgg16-based-16'], 
                in_channels=in_channels, batch_norm=batch_norm)
            self.features = en_layers

            # decoder
            de_in_channels = int(en_out_channels)
            de_layers = unets.make_decoder_layers(decoder_archs['d16-embedding'], de_in_channels, batch_norm=batch_norm)
            self.decoder = de_layers
        elif network_name == 'unet':
            # encoder
            self.encoder = unets.UNet_Encoder(input_channels=in_channels, feature_dim=num_units)

            # decoder
            self.decoder = unets.UNet_Decoder(num_encoders=1, feature_dim=num_units, coordconv=use_coordconv)
        else:
            self.fcn = getattr(resnet_dilated, network_name)(num_classes=num_units, input_channels=in_channels, pretrained=self.embedding_pretrain)
            if self.input_type == 'RGBD' and self.fusion_type != 'early':
                self.fcn_depth = getattr(resnet_dilated, network_name)(num_classes=num_units, input_channels=in_channels, pretrained=False)

        if init_weights:
            self._initialize_weights()

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, img, label, depth=None):
        if self.network_name == 'vgg':
            en = self.features(img)
        elif self.network_name == 'unet':
            en = [self.encoder(img)]

        if self.network_name == 'vgg' or self.network_name == 'unet':
            features = self.decoder(en)
        else:
            if self.input_type == 'DEPTH':
                features = self.fcn(depth)
            elif self.input_type == 'COLOR':
                features = self.fcn(img)
            elif self.input_type == 'RGBD' and self.fusion_type == 'early':
                inputs = torch.cat((img, depth), 1)
                features = self.fcn(inputs)
            else:
                features = self.fcn(img)
                features_depth = self.fcn_depth(depth)
                if self.fusion_type == 'add':
                    features = features + features_depth
                else:
                    features = torch.cat((features, features_depth), 1)

        # normalization
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        if self.training:
            loss, intra_cluster_loss, inter_cluster_loss = self.embedding_loss(features, label)
            return loss, intra_cluster_loss, inter_cluster_loss, features
        else:
            return features


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

#############################################################

def update_model(model, data):
    model_dict = model.state_dict()
    print('model keys')
    print('=================================================')
    for k, v in model_dict.items():
        print(k)
    print('=================================================')

    if data is not None:
        print('data keys')
        print('=================================================')
        data_new = data.copy()
        for k, v in data.items():
            print(k)
            # legency with the orignially trained model
            if 'module.' in k:
                data_new[k[7:]] = v
            if 'decoder.features.' in k:
                new_key = 'decoder.' + k[17:]
                data_new[new_key] = v
        print('=================================================')

        pretrained_dict = {k: v for k, v in data_new.items() if k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)


# feature embedding learning network
def seg_vgg_embedding(num_classes=2, num_units=64, data=None):
    model = SEGNET(in_channels=3, network_name='vgg', num_units=num_units)
    update_model(model, data)
    return model

def seg_unet_embedding(num_classes=2, num_units=64, data=None):
    model = SEGNET(in_channels=3, network_name='unet', num_units=num_units)
    update_model(model, data)
    return model

def seg_resnet34_8s_embedding(num_classes=2, num_units=64, data=None):
    model = SEGNET(in_channels=3, network_name='Resnet34_8s', num_units=num_units)
    update_model(model, data)
    return model

def seg_resnet34_8s_embedding_early(num_classes=2, num_units=64, data=None):
    model = SEGNET(in_channels=6, network_name='Resnet34_8s', num_units=num_units)
    update_model(model, data)
    return model

def seg_resnet50_8s_embedding(num_classes=2, num_units=64, data=None):
    model = SEGNET(in_channels=3, network_name='Resnet50_8s', num_units=num_units)
    update_model(model, data)
    return model
