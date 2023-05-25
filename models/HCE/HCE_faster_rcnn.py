# -*- coding: utf-8 -*-
# @Time : 2021/12/23 19:36
# @Author : Shen Junyong
# @File : HCE_faster_rcnn
# @Project : faster_rcnn_pytorch
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.utils import load_state_dict_from_url

from .generalize_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from .HCE_roi_head import HCE_RoIHeads
from .general_transform import GeneralizedRCNNTransform
from .backbone import resnet_fpn_backbone


__all__ = [
    "HCE_FasterRCNN", "HCE_fasterrcnn_resnet50_fpn",
]


class HCE_FasterRCNN(GeneralizedRCNN):
    def __init__(self,training_mode, test_mode,
                 backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)
            box_predictor_after_fusion = HCE_SUB_Predictor(
                representation_size,
                num_classes)

        #Hierachical Contextual RoI Feature Generation
        x_roi_pool = MultiScaleRoIAlign(
            featmap_names=['FeatureMap'],
            output_size=7,
            sampling_ratio=2)
        #去除背景类，-1
        backbone_backend = Image_Level_Neck(backbone.conv_channels,num_classes-1, backbone.out_channels)

        context_neck = HCR_ROI_Generation(x_roi_pool,backbone.conv_channels,backbone.out_channels)

        roi_heads = HCE_RoIHeads(
            #Settting
            training_mode, test_mode,
            # Box
            box_roi_pool,context_neck, box_head, box_predictor, box_predictor_after_fusion,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(HCE_FasterRCNN, self).__init__(backbone,backbone_backend, rpn, context_neck, roi_heads, transform)

class Image_Level_Neck(nn.Module):
    def __init__(self, conv_channels, num_classes, out_channels):
        super(Image_Level_Neck, self).__init__()
        self.out_channels = out_channels
        self.conv_channels = conv_channels
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(conv_channels * 2, num_classes)

    def forward(self, featuremap):
        # Image_level_categorical_embedding
        B,C,_,_ = featuremap.size()
        f_gap = self.GAP(featuremap).view(B,C)
        f_gmp = self.GMP(featuremap).view(B,C)
        f = torch.cat([f_gap,f_gmp],dim=-1)
        multi_scores = self.fc(f)
        return multi_scores

class HCR_ROI_Generation(nn.Module):
    def __init__(self, box_roi, input_channels,out_channels):
        super(HCR_ROI_Generation, self).__init__()
        self.box_roi = box_roi
        self.conv1 = nn.Conv2d(input_channels*2, out_channels, kernel_size=1)

    def forward(self, x, proposals, whole, image_shapes):
        x_instance = self.box_roi(x,proposals,image_shapes)
        x_global = self.box_roi(x,whole,image_shapes)
        x_concat = torch.cat([x_instance,x_global],dim=1)
        x = self.conv1(x_concat)
        # import matplotlib.pyplot as plt
        # from util.tools import grid_gray_image
        # import os
        # img_brain = grid_gray_image(x[0][0].unsqueeze(dim=0).detach().cpu().numpy(), 1)
        # plt.figure(figsize=(80, 80))
        # plt.imshow(img_brain, cmap='gray')
        # path = r'D:\A_Detection_Library\Baseline\Final\AMD\latest\AMD_tt_hce_HCE faster rcnn_bs1_lr0.0005_50_[20, 40]_1\VISUAL\x_global.png'
        # plt.savefig(path)
        # plt.close()
        return x

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class HCE_SUB_Predictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(HCE_SUB_Predictor, self).__init__()
        self.cls_score_context = nn.Linear(in_channels, num_classes)
        self.cls_score_fpn = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x, y):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        if y.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        scores_context = self.cls_score_fpn(x)
        scores_fpn = self.cls_score_context(y)
        scores = scores_context + scores_fpn
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def HCE_fasterrcnn_resnet50_fpn(training_mode, test_mode,pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = HCE_FasterRCNN(training_mode, test_mode,backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
