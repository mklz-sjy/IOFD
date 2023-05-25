# -*- coding: utf-8 -*-
# @Time : 2021/12/22 21:16
# @Author : Shen Junyong
# @File : evalute_brain_faster_rcnn
# @Project : faster_rcnn_pytorch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
from PIL import Image
import time
from sklearn.metrics import accuracy_score
import datetime
import util.utils as utils
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.utils.data
import random
from models.build_model_brain import create_brain_model
from datasets.Brain_tumor import build_brain_tumor
from util.tools import *
from pycocotools.coco import COCO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code for classification')

    parser.add_argument('--train_mode', help='choose one of mode: before, after or all', type=str)
    parser.add_argument('--test_mode', help='choose one of mode: before, after or all',default='all', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_classes', default=3, type=int)

    parser.add_argument('--gpu_index', default="0", type=str)
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--iou_threshold', default=0.5)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--workers', default=2)

    parser.add_argument('--output_path',
                        default="../output",
                        type=str)
    parser.add_argument('--brain_tumor_path', default="../data/brain_tvt",
                        type=str)
    #weight
    parser.add_argument('--weight_path',
                        default="../test/para/brain_best.pkl",
                        type=str)

    parser.add_argument('--final_box', default=1, type=int)#保留框个数 Number of reserved boxes

    args = parser.parse_args()
    print(f'Setting:\n{args}')

    iou_threshold = args.iou_threshold
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
    device = torch.device(args.device)
    #fix the seed for reproducibility
    seed=args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #model
    model_name = "IOFD"
    model = create_brain_model(model_name, args)
    #load weight file
    state_dict = torch.load(args.weight_path)
    model.load_state_dict(state_dict)
    model.to(device)

    output_dir = Path(args.output_path)
    output_dir = output_dir / 'evaluate'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, file)):
                os.remove('{}/{}'.format(output_dir, file))
            elif os.path.isdir(os.path.join(output_dir, file)):
                shutil.rmtree(os.path.join(output_dir, file), True)
    file_path = output_dir / 'test_result.txt'
    f = open(file_path, 'w')

    #dataset
    #test
    test_dataset, test_image_folder, test_ann_file = build_brain_tumor(image_set='test', path=args.brain_tumor_path)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, args.batch_size, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn)

    print('{}'.format(args))

    test_1_all_gts, test_2_all_gts, test_3_all_gts = get_brain_number(test_image_folder)
    print('Test status:\ttype1_image_numbers:  {},\ttype2_image_numbers:  {}\ttype3_image_numbers:  {}\n\n'.format(
        test_1_all_gts, test_2_all_gts, test_3_all_gts))
    f.write('Test status:\ttype1_image_numbers:  {},\ttype2_image_numbers:  {}\ttype3_image_numbers:  {}\n\n'.format(
        test_1_all_gts, test_2_all_gts, test_3_all_gts))

    # calculate time
    start_time = time.time()
    print('Starting Evaluating:')

    coco = COCO(test_ann_file)
    test_type_1_nd = np.empty((0, 3), dtype=np.float32)
    test_type_2_nd = np.empty((0, 3), dtype=np.float32)
    test_type_3_nd = np.empty((0, 3), dtype=np.float32)

    all_num = 0
    all_right_num = 0
    all_iou_error_num = 0
    all_class_error_num = 0
    all_both_error_num = 0

    test_labels = []
    test_preds = []

    for s, data in enumerate(test_dataloader):
        model.eval()
        inputs, labels = data
        # labels is a tuple, boxes,labels,image_id,orig_size,size
        inputs = list(input.to(device) for input in inputs)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]  # [{},{}...]

        outputs,i= model(inputs)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]  # [{},{}...]
        # outputs:[{"boxes":{},"labels":{}, "scores":{}}]  [100,4],[100],[100] 100 predictions
        if args.final_box == 1:
            for l, p in zip(labels, outputs):
                test_labels.append(l['labels'].item())
                try:
                    test_preds.append(p['labels'].item())
                except ValueError:
                    if l['labels'].item() == 1:
                        test_preds.append(2)
                    elif l['labels'].item() == 2:
                        test_preds.append(3)
                    else:
                        test_preds.append(1)

        print('Test:[{}/{}]'.format(s, len(test_dataloader)))
        for label, output in zip(labels, outputs):  # 解决batch_size>1
            img_id = label['image_id'].item()
            img_path = coco.loadImgs(img_id)[0]['file_name']
            f_img = h5py.File(img_path, 'r')
            img = np.asarray(f_img['cjdata/image'])
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = img * 255
            origin_img = Image.fromarray(img).convert('L')
            people_id = coco.loadImgs(img_id)[0]['license']
            people_name = people_id
            img_index = str(img_id)
            scores, ious, error = plot_iou_results(label, output, origin_img, args.output_path, people_name, img_index)
            right_num, iou_error, class_error, both_error = error
            all_num = all_num + right_num + iou_error + class_error + both_error
            all_right_num += right_num
            all_iou_error_num += iou_error
            all_class_error_num += class_error
            all_both_error_num += both_error


            each_1_nd = get_each_class_image_nd(output, label, iou_threshold, 1)  # scores,tp,fp
            test_type_1_nd = np.append(test_type_1_nd, each_1_nd, axis=0)

            each_2_nd = get_each_class_image_nd(output, label, iou_threshold, 2)
            test_type_2_nd = np.append(test_type_2_nd, each_2_nd, axis=0)

            each_3_nd = get_each_class_image_nd(output, label, iou_threshold, 3)
            test_type_3_nd = np.append(test_type_3_nd, each_3_nd, axis=0)

    # type_1
    precion_1_nd, recall_1_nd = get_pre_rec_nd(test_type_1_nd, test_1_all_gts)
    test_ap_1 = get_ap(recall_1_nd, precion_1_nd, use_07_metric=False)
    # type_2
    precion_2_nd, recall_2_nd = get_pre_rec_nd(test_type_2_nd, test_2_all_gts)
    test_ap_2 = get_ap(recall_2_nd, precion_2_nd, use_07_metric=False)
    #type_3
    precion_3_nd, recall_3_nd = get_pre_rec_nd(test_type_3_nd, test_3_all_gts)
    test_ap_3 = get_ap(recall_3_nd, precion_3_nd, use_07_metric=False)

    test_map = (test_ap_1 + test_ap_2 + test_ap_3) / 3

    test_all_gts = test_image_num = test_1_all_gts + test_2_all_gts + test_3_all_gts

    test_type_all_nd = np.append(test_type_1_nd, test_type_2_nd, axis=0)
    test_type_all_nd = np.append(test_type_all_nd, test_type_3_nd, axis=0)
    test_fps, test_sensitivity = get_fps_sen_nd(test_type_all_nd, test_image_num, test_all_gts)
    index = index_number(test_fps, 0.5)
    f.write('fps=0.5\t sensitivity={}\n'.format(test_sensitivity[index]))
    # 1
    index = index_number(test_fps, 1)
    f.write('fps=1\t sensitivity={}\n'.format(test_sensitivity[index]))
    # 2
    index = index_number(test_fps, 2)
    f.write('fps=2\t sensitivity={}\n'.format(test_sensitivity[index]))
    # 4
    index = index_number(test_fps, 4)
    f.write('fps=4\t sensitivity={}\n'.format(test_sensitivity[index]))
    # 8
    index = index_number(test_fps, 8)
    f.write('fps=8\t sensitivity={}\n'.format(test_sensitivity[index]))
    # 16
    index = index_number(test_fps, 16)
    f.write('fps=16\t sensitivity={}\n'.format(test_sensitivity[index]))

    print('test metrics:\tap_1:  {},\tap_2:  {},\tap_3:  {},\tmap:  {}\n'.format(test_ap_1,test_ap_2,test_ap_3,test_map))
    f.write(
        'test metrics:\tap_1:  {},\tap_2:  {},\tap_3:  {},\tmap:  {}\n'.format(test_ap_1,test_ap_2,test_ap_3,test_map))

    plot_x_y(test_fps, test_sensitivity, 'Average number of false positives per image', 'Sensitivity', 'FROC performence',
             output_dir, epoch='test')

    right_ratio = all_right_num / all_num
    iou_error_ratio = all_iou_error_num / all_num
    class_error_ratio = all_class_error_num / all_num
    both_error_ratio = all_both_error_num / all_num
    if args.final_box==1:
        test_acc = accuracy_score(test_labels, test_preds)
        f.write('test_acc:{}\n'.format(test_acc))
    f.write('test:{},{},{},{},{}\n right_ratio:{}, iou_error_ratio:{}, class_error_ratio:{}, both_error_ratio:{}'.format(
            all_num, all_right_num, all_iou_error_num, all_class_error_num, all_both_error_num,
            right_ratio, iou_error_ratio, class_error_ratio, both_error_ratio
            ))

    f.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))