# -*- coding: utf-8 -*-
# @Time : 2021/12/22 17:39
# @Author : Shen Junyong
# @File : build_model_brain
# @Project : faster_rcnn_pytorch
import torch.nn as nn
from util.calculate_mean_std import get_brain_dataset_mean_std

def create_brain_model(model_name,args):
    if model_name == 'FasterRCNN':
        mean, std = get_brain_dataset_mean_std(args)
        from models.faster_rcnn.faster_rcnn import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(num_classes=args.num_classes+1,image_mean=[mean, mean, mean],
                                        image_std=[std, std, std],box_detections_per_img=args.final_box)
        model.roi_heads.box_predictor.cls_score = nn.Linear(1024, args.num_classes+1)
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4*(args.num_classes+1))

    elif model_name == 'MSB':
        mean, std = get_brain_dataset_mean_std(args)
        from models.faster_rcnn.faster_rcnn import fasterrcnn_resnet50_MSB
        model = fasterrcnn_resnet50_MSB(num_classes=args.num_classes + 1, image_mean=[mean, mean, mean],
                                        image_std=[std, std, std], box_detections_per_img=args.final_box)
        model.roi_heads.box_predictor.cls_score = nn.Linear(1024, args.num_classes + 1)
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4 * (args.num_classes + 1))

    elif model_name == 'HCE':
        mean, std = get_brain_dataset_mean_std(args)
        from models.HCE.HCE_faster_rcnn import HCE_fasterrcnn_resnet50_fpn
        model = HCE_fasterrcnn_resnet50_fpn(args.train_mode, args.test_mode,num_classes=args.num_classes + 1, image_mean=[mean, mean, mean],
                                        image_std=[std, std, std], box_detections_per_img=args.final_box)

    elif model_name == 'IOFD':
        mean, std = get_brain_dataset_mean_std(args)
        from models.IOFD.Decouple_faster_rcnn import Decouple_fasterrcnn_resnet50_fpn
        model = Decouple_fasterrcnn_resnet50_fpn(num_classes=args.num_classes + 1, image_mean=[mean, mean, mean],
                                        image_std=[std, std, std], box_detections_per_img=args.final_box)
    else:
        assert False,'Model Name Error'
    return model