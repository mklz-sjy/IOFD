# -*- coding: utf-8 -*-
# @Time : 2021/12/23 19:44
# @Author : Shen Junyong
# @File : backbone
# @Project : faster_rcnn_pytorch
from collections import OrderedDict
from torch import nn
import torch
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet


class BackboneWithFPN(nn.Module):
    def __init__(self, conv_channels, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.conv_channels = conv_channels
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3,padding=1)
    def forward(self, x):
        x = self.body(x)
        last_featuremap = x['3']
        featuremap = self.conv3(last_featuremap)
        x = self.fpn(x)
        return x, featuremap

def resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    conv_channels = in_channels_stage2*8
    return BackboneWithFPN(conv_channels, backbone, return_layers, in_channels_list, out_channels)
