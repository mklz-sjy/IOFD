# -*- coding: utf-8 -*-
# @Time : 2021/12/22 17:31
# @Author : Shen Junyong
# @File : brain_main_templex
# @Project : faster_rcnn_pytorch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import shutil
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code for classification')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--epochs', default=50, type=int)
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

    parser.add_argument('--final_box', default=1, type=int)#保留框个数 Number of reserved boxes
    args = parser.parse_args()
    print(f'Setting:\n{args}')


    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
    device = torch.device(args.device)
    #fix the seed for reproducibility
    seed=args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #model
    model_name = 'MSB'
    model = create_brain_model(model_name, args)

    args.task_name = '{}/BrainTumor Model_{} Bs_{} Lr_{} Epoch_{} LrDrop_{}'.format(args.output_path, model_name, args.batch_size,args.lr,args.epochs,args.lr_drop)
    if not os.path.isdir(args.task_name):
        os.makedirs(args.task_name)
    else:
        for file in os.listdir(args.task_name):
            if os.path.isfile(os.path.join(args.task_name, file)):
                os.remove('{}/{}'.format(args.task_name, file))
            elif os.path.isdir(os.path.join(args.task_name, file)):
                shutil.rmtree(os.path.join(args.task_name, file), True)

    model.to(device)

    # log
    log_path = os.path.join(args.task_name, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    else:
        for file in os.listdir(log_path):
            if os.path.isfile(os.path.join(log_path, file)):
                os.remove('{}/{}'.format(log_path, file))
            elif os.path.isdir(os.path.join(log_path, file)):
                shutil.rmtree(os.path.join(log_path, file), True)
    writer = SummaryWriter(log_dir=log_path)

    #optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    #lr_scheduler
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,args.lr_drop)

    #dataset
    #train
    train_dataset,train_image_folder,train_ann_file=build_brain_tumor(image_set='train',path=args.brain_tumor_path)
    #valid
    valid_dataset,val_image_folder,val_ann_file=build_brain_tumor(image_set='val',path=args.brain_tumor_path)
    #test
    test_dataset, test_image_folder, test_ann_file = build_brain_tumor(image_set='test', path=args.brain_tumor_path)

    #Sampler
    train_sampler = torch.utils.data.RandomSampler(train_dataset)#generate the indexs list
    train_batch_sampler=torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)#pack the indexs list to a group

    valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    valid_batch_sampler = torch.utils.data.BatchSampler(valid_sampler, args.batch_size, drop_last=False)

    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, args.batch_size, drop_last=False)
    #dataloader
    train_dataloader=torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,collate_fn=utils.collate_fn)

    valid_dataloader=torch.utils.data.DataLoader(
        valid_dataset, batch_sampler=valid_batch_sampler, num_workers=args.workers,collate_fn=utils.collate_fn)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn)

    file_path=os.path.join(args.task_name,'result.txt')
    f=open(file_path,'w')
    f.write('{}'.format(args))

    train_1_all_gts, train_2_all_gts, train_3_all_gts = get_brain_number(train_image_folder)
    f.write('Train status:\ttype1_image_numbers:  {},\ttype2_image_numbers:  {}\ttype3_image_numbers:  {}\n'.format(train_1_all_gts,
                                                                                          train_2_all_gts, train_3_all_gts))

    valid_1_all_gts, valid_2_all_gts, valid_3_all_gts = get_brain_number(val_image_folder)
    f.write('Valid status:\ttype1_image_numbers:  {},\ttype2_image_numbers:  {}\ttype3_image_numbers:  {}\n\n'.format(valid_1_all_gts, valid_2_all_gts, valid_3_all_gts))

    test_1_all_gts, test_2_all_gts, test_3_all_gts = get_brain_number(test_image_folder)
    f.write('Test status:\ttype1_image_numbers:  {},\ttype2_image_numbers:  {}\ttype3_image_numbers:  {}\n\n'.format(
        test_1_all_gts, test_2_all_gts, test_3_all_gts))

    # calculate time
    start_time = time.time()
    print('Starting Train:')

    best_map = 0

    for epoch in range(args.epochs):
        train_losses=0
        train_total_loss_classifier=0
        train_total_loss_boxreg = 0
        train_total_loss_objectness = 0
        train_total_loss_rpnboxreg = 0

        for i, data in enumerate(train_dataloader):
            model.train()

            inputs, labels =data
            inputs=list(input.to(device) for input in inputs)
            labels=[{k: v.to(device) for k,v in t.items()} for t in labels]
            losses_dict= model(inputs,labels)

            total_losses = losses_dict['loss_classifier'] + losses_dict['loss_box_reg'] + losses_dict['loss_objectness'] + losses_dict['loss_rpn_box_reg']

            optimizer.zero_grad()

            total_losses.backward()

            optimizer.step()

            train_losses+=total_losses.item()
            train_total_loss_classifier += losses_dict['loss_classifier'].item()
            train_total_loss_boxreg += losses_dict['loss_box_reg'].item()
            train_total_loss_objectness += losses_dict['loss_objectness'].item()
            train_total_loss_rpnboxreg += losses_dict['loss_rpn_box_reg'].item()

            print('Train: [{}:{}/{}]: train_losses：{}, loss_classifier:{}, loss_box_reg:{}, loss_objectness:{}, '
                  'loss_rpn_box_reg:{}'.format(epoch,i,len(train_dataloader),
                                               total_losses.item(), losses_dict['loss_classifier'].item(), losses_dict["loss_box_reg"].item(),
                                                                                                                 losses_dict["loss_objectness"].item(),losses_dict["loss_rpn_box_reg"].item()))
        lr_scheduler.step()

        train_average_loss=train_losses/(i+1)
        train_average_loss_classifier = train_total_loss_classifier/(i+1)
        train_average_loss_boxreg = train_total_loss_boxreg/(i+1)
        train_average_loss_objectness = train_total_loss_objectness/(i+1)
        train_average_loss_rpnboxreg = train_total_loss_rpnboxreg/(i+1)


        f.write('Epoch_{},\ttrain_average_loss:  {},\ttrain_average_loss_classifier:  {},\ttrain_average_loss_boxreg:  {},\ttrain_average_loss_objectness:  {},\ttrain_average_loss_rpnboxreg:  {}\n'.format(
            epoch,train_average_loss,train_average_loss_classifier,train_average_loss_boxreg,train_average_loss_objectness,train_average_loss_rpnboxreg))
        writer.add_scalar('train_loss', train_average_loss, epoch)
        writer.add_scalar('train_average_loss_classifier', train_average_loss_classifier, epoch)
        writer.add_scalar('train_average_loss_boxreg', train_average_loss_boxreg, epoch)
        writer.add_scalar('train_average_loss_objectness', train_average_loss_objectness, epoch)
        writer.add_scalar('train_average_loss_rpnboxreg', train_average_loss_rpnboxreg, epoch)

        #valid
        iou_threshold=args.iou_threshold
        type_1_nd=np.empty((0,3),dtype=np.float32)
        type_2_nd=np.empty((0,3),dtype=np.float32)
        type_3_nd=np.empty((0,3),dtype=np.float32)

        valid_labels = []
        valid_preds = []
        for j, data in enumerate(valid_dataloader):
            model.eval()
            inputs, labels = data
            #labels is a tuple, boxes,labels,image_id,orig_size,size
            inputs = list(input.to(device) for input in inputs)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]#[{},{}...]
            outputs = model(inputs)
            outputs = [{k: v.to(device) for k,v in t.items()} for t in outputs]#[{},{}...]
            #outputs:[{"boxes":{},"labels":{}, "scores":{}}]  [100,4],[100],[100] 100 predictions

            for l, p in zip(labels, outputs):
                valid_labels.append(l['labels'].item())
                try:
                    valid_preds.append(p['labels'].item())
                except ValueError:
                    if l['labels'].item() == 1:
                        valid_preds.append(2)
                    elif l['labels'].item()==2:
                        valid_preds.append(3)
                    else:
                        valid_preds.append(1)

            print('Valid:[{}:{}/{}]'.format(epoch,j, len(valid_dataloader)))
            for label,output in zip(labels,outputs):#解决batch_size>1

                each_1_nd = get_each_class_image_nd(output, label, iou_threshold,1)  # scores,tp,fp
                type_1_nd = np.append(type_1_nd, each_1_nd, axis=0)

                each_2_nd = get_each_class_image_nd(output, label, iou_threshold, 2)
                type_2_nd = np.append(type_2_nd, each_2_nd, axis=0)

                each_3_nd = get_each_class_image_nd(output, label, iou_threshold, 3)
                type_3_nd = np.append(type_3_nd, each_3_nd, axis=0)

        # evaluation metrics
        # each category;each iou and each class threshold
        #type_1
        precion_nd, recall_nd=get_pre_rec_nd(type_1_nd, valid_1_all_gts)
        ap_1=get_ap(recall_nd,precion_nd,use_07_metric=False)
        # type_2
        precion_2_nd, recall_2_nd = get_pre_rec_nd(type_2_nd, valid_2_all_gts)
        ap_2 = get_ap(recall_2_nd, precion_2_nd, use_07_metric=False)
        # type_3
        precion_3_nd, recall_3_nd = get_pre_rec_nd(type_3_nd, valid_3_all_gts)
        ap_3 = get_ap(recall_3_nd, precion_3_nd, use_07_metric=False)

        map=(ap_1+ap_2+ap_3)/3
        #get froc curve
        valid_all_gts = valid_image_num = valid_1_all_gts + valid_2_all_gts + valid_3_all_gts
        type_all_nd = np.append(type_1_nd, type_2_nd, axis=0)
        type_all_nd = np.append(type_all_nd, type_3_nd, axis=0)
        fps, sensitivity = get_fps_sen_nd(type_all_nd, valid_image_num, valid_all_gts)
        # index = index_number(fps, 1)
        # t_sensitivity = sensitivity[index]
        # f.write('fps:\t{}\n sensitivity:\t{}\n'.format(fps,t_sensitivity))
        valid_acc = accuracy_score(valid_labels, valid_preds)

        #valid metrics curve
        writer.add_scalar('valid_acc', valid_acc, epoch)
        # writer.add_scalar('valid_sensitivity', t_sensitivity, epoch)

        writer.add_scalar('valid_ap_1', ap_1, epoch)
        writer.add_scalar('valid_ap_2', ap_2, epoch)
        writer.add_scalar('valid_ap_3', ap_3, epoch)
        writer.add_scalar('valid_map', map, epoch)

        f.write('valid metrics:\tap_1:  {},\tap_2:  {},\tap_3:  {},\tmap:  {}\n\n'.format(ap_1,ap_2,ap_3,map))
        print('valid metrics:\tap_1:  {},\tap_2:  {},\tap_3:  {},\tmap:  {}\n\n'.format(ap_1, ap_2, ap_3, map))

        #save model
        if map > best_map:

            test_type_1_nd = np.empty((0, 3), dtype=np.float32)
            test_type_2_nd = np.empty((0, 3), dtype=np.float32)
            test_type_3_nd = np.empty((0, 3), dtype=np.float32)

            test_labels = []
            test_preds = []
            for s, data in enumerate(test_dataloader):
                model.eval()
                inputs, labels = data
                # labels is a tuple, boxes,labels,image_id,orig_size,size
                inputs = list(input.to(device) for input in inputs)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]  # [{},{}...]

                outputs = model(inputs)
                outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]  # [{},{}...]
                # outputs:[{"boxes":{},"labels":{}, "scores":{}}]  [100,4],[100],[100] 100 predictions

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

                print('Test:[{}:{}/{}]'.format(epoch, s, len(test_dataloader)))
                for label, output in zip(labels, outputs):  # 解决batch_size>1

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
            # index = index_number(test_fps, 1)
            # test_sensitivity = test_sensitivity[index]
            test_acc = accuracy_score(test_labels, test_preds)
            f.write('test metrics:\tap_1:  {},\tap_2:  {},\tap_3:  {},\tmap:  {}\ttest_acc:{}\n\n'.format(test_ap_1,
                                                                                                            test_ap_2,
                                                                                                            test_ap_3,
                                                                                                            test_map,
                                                                                                            test_acc))
            best_map = map

            checkpoint = str(epoch) + "_best_" + str(map)+'_'+str(valid_acc)+'_'+str(test_acc)+'_'+str(test_map)
            torch.save(model.state_dict(), '{}/net_{}.pkl'.format(args.task_name, checkpoint))

    f.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))