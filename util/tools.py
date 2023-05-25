# -*- coding: utf-8 -*-
# @Time : 2021/7/30 11:12
# @Author : Shen Junyong
# @File : tools
# @Project : code_for_classification_detection
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torch
import matplotlib
from decimal import *
import re
import h5py

def get_brain_dataset_mean_std(args):
    train_folder = os.path.join(args.brain_tumor_path,'train')
    mean = 0
    std = 0
    num = 0
    for img in os.listdir(train_folder):
        num+=1
        f = h5py.File(os.path.join(train_folder,img), 'r')
        image_nd = np.asarray(f['cjdata/image'])
        image_nd = image_nd.astype(np.float32) / 255
        mean+=image_nd.mean()
        std+=image_nd.std()
    mean = round(mean / num, 3)
    std = round(std / num, 3)
    print(mean)#0.20367678290989308
    print(std)#0.22138273613276166
    return mean,std


def plot_results(label,output,pil_img,output_dir,people_name,img_index):
    matplotlib.use('AGG')
    score_list = []
    iou_list = []
    import matplotlib.pyplot as plt
    gt_boxes=label['boxes'].detach().cpu().numpy().tolist()[0]
    gt_label = label['labels'].detach().cpu().numpy()[0]

    iou_error = 0
    both_error = 0
    class_error = 0
    right_num = 0

    print(people_name,img_index,gt_boxes)
    print(output['boxes'])
    CLASSES = [
        'type I', 'type II'
    ]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes
    ax.add_patch(plt.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                               fill=False, color='red', linewidth=3))
    text = f'{CLASSES[gt_label - 1]}'
    # ax.text(gt_xmin, gt_ymax+25, text, fontsize=30,
    #         bbox=dict(facecolor='red', alpha=0.5))
    ax.text(gt_xmax, gt_ymax, text, fontsize=15,
            bbox=dict(facecolor='red', alpha=0.5))

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()

    index = np.argsort(-scores)

    for n in range(scores.shape[0]):
        i = index[n]
        score = scores[i]
        xmin, ymin, xmax, ymax = boxes[i, :]
        label = labels[i]

        # calculate iou
        x1 = max(xmin, gt_xmin)
        y1 = max(ymin, gt_ymin)
        x2 = min(xmax, gt_xmax)
        y2 = min(ymax, gt_ymax)
        if x2 - x1 < 0 or y2 - y1 < 0:
            iou = 0
        else:
            overlap = (x2 - x1) * (y2 - y1)
            union = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) + (xmax - xmin) * (ymax - ymin) - overlap
            iou = overlap / union

        if iou > 0.4 and label == gt_label:  # 正确预测
            c = 'green'
            right_num += 1
        elif iou < 0.4 and label == gt_label:  # 交并比低
            c = 'yellow'
            iou_error += 1
        elif iou > 0.4 and label != gt_label:  # 类别判断错误
            c = 'cyan'
            class_error += 1
        else:  # 均错
            c = 'blue'
            both_error += 1

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[label-1]}'
        # text = f'{CLASSES[label - 1]}: {score:0.3f}'
        # ax.text(xmin, ymin-10, text, fontsize=30, color='lime',
        #         bbox=dict(facecolor=c, alpha=0.5))
        #text = f'{CLASSES[label - 1]}: {score:0.2f},{iou:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor=c, alpha=0.5))

        score_list.append(score)
        iou_list.append(iou)

    plt.axis('off')  # 关闭坐标轴
    new_output_dir = Path(output_dir) / people_name
    if not os.path.isdir(new_output_dir):
        os.makedirs(new_output_dir)
    image_out = new_output_dir / img_index
    plt.savefig(image_out)
    plt.close()

    error = [right_num, iou_error, class_error, both_error]
    return score_list, iou_list, error

def plot_prediction(label,pil_img,name,img_index,output,output_dir):
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt

    gt_boxes = label['boxes'].detach().cpu().numpy().tolist()[0]
    gt_label = label['labels'].detach().cpu().numpy()[0]

    CLASSES = [
        'meningioma', 'glioma', 'pituitary tumor'
    ]
    # CLASSES = [
    #     'type I', 'type II'
    # ]

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    print(boxes)

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes
    ax.add_patch(plt.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                               fill=False, color='red', linewidth=3))
    text = f'{CLASSES[gt_label - 1]}'
    ax.text(gt_xmin, gt_ymax+25, text, fontsize=30,
            bbox=dict(facecolor='red', alpha=0.5))
    # ax.text(gt_xmax, gt_ymax, text, fontsize=15,
    #         bbox=dict(facecolor='red', alpha=0.5))

    if scores.shape[0]>0:
        index = np.argsort(-scores)
        # plt.figure(figsize=(16, 10))
        # plt.imshow(pil_img)
        # ax = plt.gca()
        for n in range(scores.shape[0]):
            i = index[n]
            score = scores[i]
            xmin, ymin, xmax, ymax = boxes[i, :]
            label = labels[i]

            # calculate iou
            x1 = max(xmin, gt_xmin)
            y1 = max(ymin, gt_ymin)
            x2 = min(xmax, gt_xmax)
            y2 = min(ymax, gt_ymax)
            if x2 - x1 < 0 or y2 - y1 < 0:
                iou = 0
            else:
                overlap = (x2 - x1) * (y2 - y1)
                union = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) + (xmax - xmin) * (ymax - ymin) - overlap
                iou = overlap / union

            if iou > 0.5 and label == gt_label:  # 正确预测
                c = 'green'
            elif iou < 0.5 and label == gt_label:  # 交并比低
                c = 'yellow'
            elif iou > 0.5 and label != gt_label:  # 类别判断错误
                c = 'cyan'
            else:  # 均错
                c = 'blue'

            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'{CLASSES[label - 1]}: {score:0.3f}'
            ax.text(xmin, ymin - 10, text, fontsize=30, color='yellow',
                    bbox=dict(facecolor=c, alpha=0.5))

            # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
            #                            fill=False, color='lime', linewidth=3))
            # text = f'{CLASSES[label - 1]}: {score:0.3f}'
            # ax.text(xmin, ymin-10, text, fontsize=30, color='lime',
            #         bbox=dict(facecolor='green', alpha=0.5))

        plt.axis('off')  # 关闭坐标轴
        new_output_dir = Path(output_dir) / name
        if not os.path.isdir(new_output_dir):
            os.makedirs(new_output_dir)
        image_out = new_output_dir / img_index
        plt.savefig(image_out)
        plt.close()

def save2json(output,output_dir,name):
    import json

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()

    if scores.shape[0]>0:
        index = np.argsort(-scores)
        for n in range(scores.shape[0]):
            i = index[n]
            result = {}
            result["x_min"]=round(boxes[i, 0])
            result['y_min']=round(boxes[i, 1])
            result['x_max']=round(boxes[i, 2])
            result['y_max']=round(boxes[i, 3])

            # 生成json文件
            with open(os.path.join(output_dir,"{}.json".format(name[:-5])), "w") as f:
                json.dump(result, f)
                print("已生成{}.json文件".format(name[:-5]))


def get_type_images(image_folder,label_xlxs_path):
    gt_type_1_numbers=0
    gt_type_2_numbers=0
    xlxs_data=pd.read_excel(label_xlxs_path)
    for people in image_folder.iterdir():
        type=xlxs_data.loc[xlxs_data['name_id']==people.name,'specialist_prior'].values[0]
        if type =='I':
            gt_type_1_numbers+=xlxs_data.loc[xlxs_data['name_id']==people.name,'image_numbers'].values[0]
        elif type =='II':
            gt_type_2_numbers+=xlxs_data.loc[xlxs_data['name_id']==people.name,'image_numbers'].values[0]
        else:
            assert False, 'no this people data'
    return gt_type_1_numbers,gt_type_2_numbers

def get_new_label_type_images(image_folder):
    gt_type_1_numbers=0
    gt_type_2_numbers=0
    for people in image_folder.iterdir():
        type = int(re.findall(r"\d+", people.name)[-1])
        people_path = image_folder / people
        img_list = [i.name for i in people_path.iterdir()]
        img_list.remove('Label')
        img_list.remove('Result')
        if type == 0:
            gt_type_1_numbers+=len(img_list)
        elif type == 1:
            gt_type_2_numbers+=len(img_list)
        else:
            assert False, 'no this people data'
    return gt_type_1_numbers,gt_type_2_numbers

def get_ap(rec,prec,use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:#np.sum(bool list)=the number of true
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_lesion_nd(output,label,iou_threshold):
    each_image_nd = np.empty([0, 3], dtype=np.float32)

    outpredicions_boxes = output['boxes'].detach().cpu().numpy()
    outpredicions_scores = output['scores'].detach().cpu().numpy()
    outpredicions_labels = output['labels'].detach().cpu().numpy()

    sort_type_index = np.argsort(-outpredicions_scores)

    #GT
    gt_boxes = label['boxes'].detach().cpu().numpy()

    if gt_boxes.shape[0] == 1:
        GT_1_mismatch = True
    else:
        GT_1_mismatch = False

    for n in range(outpredicions_boxes.shape[0]):  # 遍历每个框
        # 这里只针对数据集中只有一个框的保证GT不重复匹配
        i = sort_type_index[n]
        each_proposal_nd = np.zeros([1,3], dtype=np.float32)

        if GT_1_mismatch:
            xmin_, ymin_, xmax_, ymax_ = gt_boxes[0]
            xmin, ymin, xmax, ymax = outpredicions_boxes[i]
            x1=max(xmin,xmin_)
            y1=max(ymin,ymin_)
            x2=min(xmax,xmax_)
            y2=min(ymax,ymax_)
            if x2-x1<0 or y2-y1<0:
                iou=0
            else:
                overlap=(x2-x1)*(y2-y1)
                union=(xmax_-xmin_)*(ymax_-ymin_)+(xmax-xmin)*(ymax-ymin)-overlap
                iou=overlap/union
            if iou >= iou_threshold:
                each_proposal_nd[:,1] = 1 #tp:correct detection
                GT_1_mismatch = False
            else:
                each_proposal_nd[:,2] = 1 #fp:incorrect detection
        else:
            each_proposal_nd[:,2] = 1 #fp

        each_proposal_nd[:,0] = outpredicions_scores[i]#score

        each_image_nd = np.\
            append(each_image_nd,each_proposal_nd,axis=0)

    return each_image_nd#scores,tp,fp

def get_each_class_image_nd(output,label,iou_threshold,i):
    each_image_type_nd = np.empty([0, 3], dtype=np.float32)

    outpredicions_boxes = output['boxes'].detach().cpu().numpy()
    outpredicions_scores = output['scores'].detach().cpu().numpy()
    outpredicions_labels = output['labels'].detach().cpu().numpy()

    type_index = outpredicions_labels == i

    outpredicions_type_boxes = outpredicions_boxes[type_index,:]
    outpredicions_type_scores = outpredicions_scores[type_index]

    sort_type_index = np.argsort(-outpredicions_type_scores)

    #GT
    gt_labels = label['labels'].detach().cpu().numpy()
    gt_boxes = label['boxes'].detach().cpu().numpy()

    gt_index = gt_labels == i
    gt_labels = gt_labels[gt_index]
    gt_boxes = gt_boxes[gt_index,:]

    if gt_boxes.shape[0] == 1:
        GT_mismatch = True
    else:
        GT_mismatch = False

    for n in range(outpredicions_type_boxes.shape[0]):  # 遍历每个框
        # 这里只针对数据集中只有一个框的保证GT不重复匹配
        i = sort_type_index[n]
        each_proposal_nd = np.zeros([1,3], dtype=np.float32)

        if GT_mismatch:
            xmin_, ymin_, xmax_, ymax_ = gt_boxes[0]
            xmin, ymin, xmax, ymax = outpredicions_type_boxes[i]
            x1=max(xmin,xmin_)
            y1=max(ymin,ymin_)
            x2=min(xmax,xmax_)
            y2=min(ymax,ymax_)
            if x2-x1<0 or y2-y1<0:
                iou=0
            else:
                overlap=(x2-x1)*(y2-y1)
                union=(xmax_-xmin_)*(ymax_-ymin_)+(xmax-xmin)*(ymax-ymin)-overlap
                iou=overlap/union
            if iou >= iou_threshold:
                each_proposal_nd[:,1] = 1 #tp:correct detection
                GT_mismatch = False
            else:
                each_proposal_nd[:,2] = 1 #fp:incorrect detection
        else:
            each_proposal_nd[:,2] = 1 #fp

        each_proposal_nd[:,0] = outpredicions_type_scores[i]#score
        each_image_type_nd = np.append(each_image_type_nd,each_proposal_nd,axis=0)
    return each_image_type_nd#scores,tp,fp

def get_each_image_nd(output,label,iou_threshold):
    each_image_type_1_nd = np.empty([0, 3], dtype=np.float32)
    each_image_type_2_nd = np.empty([0, 3], dtype=np.float32)

    outpredicions_boxes = output['boxes'].detach().cpu().numpy()
    outpredicions_scores = output['scores'].detach().cpu().numpy()
    outpredicions_labels = output['labels'].detach().cpu().numpy()

    type_1_index = outpredicions_labels == 1
    type_2_index = outpredicions_labels == 2

    outpredicions_type_1_boxes = outpredicions_boxes[type_1_index,:]
    outpredicions_type_1_scores = outpredicions_scores[type_1_index]

    outpredicions_type_2_boxes = outpredicions_boxes[type_2_index,:]
    outpredicions_type_2_scores = outpredicions_scores[type_2_index]

    sort_type_1_index = np.argsort(-outpredicions_type_1_scores)
    sort_type_2_index = np.argsort(-outpredicions_type_2_scores)

    #GT
    gt_labels = label['labels'].detach().cpu().numpy()
    gt_boxes = label['boxes'].detach().cpu().numpy()

    gt_1_index = gt_labels == 1
    gt_1_labels = gt_labels[gt_1_index]
    gt_1_boxes = gt_boxes[gt_1_index,:]
    #xmin_,ymin_,xmax_,ymax_ = label['boxes'].detach().cpu().numpy()[0] #* np.array([w,h,w,h],dtype=np.float32)

    gt_2_index = gt_labels == 2
    gt_2_labels = gt_labels[gt_2_index]
    gt_2_boxes = gt_boxes[gt_2_index,:]

    #type I
    if gt_1_boxes.shape[0] == 1:
        GT_1_mismatch = True
    else:
        GT_1_mismatch = False

    for n in range(outpredicions_type_1_boxes.shape[0]):  # 遍历每个框
        # 这里只针对数据集中只有一个框的保证GT不重复匹配
        i = sort_type_1_index[n]
        each_proposal_nd = np.zeros([1,3], dtype=np.float32)

        if GT_1_mismatch:
            xmin_, ymin_, xmax_, ymax_ = gt_1_boxes[0]
            xmin, ymin, xmax, ymax = outpredicions_type_1_boxes[i]
            x1=max(xmin,xmin_)
            y1=max(ymin,ymin_)
            x2=min(xmax,xmax_)
            y2=min(ymax,ymax_)
            if x2-x1<0 or y2-y1<0:
                iou=0
            else:
                overlap=(x2-x1)*(y2-y1)
                union=(xmax_-xmin_)*(ymax_-ymin_)+(xmax-xmin)*(ymax-ymin)-overlap
                iou=overlap/union
            if iou >= iou_threshold:
                each_proposal_nd[:,1] = 1 #tp:correct detection
                GT_1_mismatch = False
            else:
                each_proposal_nd[:,2] = 1 #fp:incorrect detection
        else:
            each_proposal_nd[:,2] = 1 #fp

        each_proposal_nd[:,0] = outpredicions_type_1_scores[i]#score

        each_image_type_1_nd = np.append(each_image_type_1_nd,each_proposal_nd,axis=0)

    # type II
    if gt_2_boxes.shape[0] == 1:
        GT_2_mismatch = True
    else:
        GT_2_mismatch = False

    for n in range(outpredicions_type_2_boxes.shape[0]):  # 遍历每个框
        # 这里只针对数据集中只有一个框的保证GT不重复匹配
        i = sort_type_2_index[n]
        each_proposal_nd = np.zeros([1, 3], dtype=np.float32)

        if GT_2_mismatch:
            xmin_, ymin_, xmax_, ymax_ = gt_2_boxes[0]
            xmin, ymin, xmax, ymax = outpredicions_type_2_boxes[i]
            x1 = max(xmin, xmin_)
            y1 = max(ymin, ymin_)
            x2 = min(xmax, xmax_)
            y2 = min(ymax, ymax_)
            if x2 - x1 < 0 or y2 - y1 < 0:
                iou = 0
            else:
                overlap = (x2 - x1) * (y2 - y1)
                union = (xmax_ - xmin_) * (ymax_ - ymin_) + (xmax - xmin) * (ymax - ymin) - overlap
                iou = overlap / union

            if iou >= iou_threshold:
                each_proposal_nd[:, 1] = 1  # tp:correct detection
                GT_2_mismatch = False
            else:
                each_proposal_nd[:, 2] = 1  # fp:incorrect detection
        else:
            each_proposal_nd[:, 2] = 1  # fp

        each_proposal_nd[:, 0] = outpredicions_type_2_scores[i]  # score

        each_image_type_2_nd = np.append(each_image_type_2_nd, each_proposal_nd, axis=0)

    return each_image_type_1_nd, each_image_type_2_nd#scores,tp,fp

def get_pre_rec_nd(type_nd,type_all_gts):
    sort_index = np.argsort(-type_nd[:, 0])#降序排列
    final_type_nd = np.zeros((len(sort_index), 7), dtype=np.float32)#scores,tp,fp,sum tp,sum fp,pre,recall
    for i in range(len(sort_index)):
        # scores,tp,fp
        final_type_nd[i, :3] = type_nd[sort_index[i], :3]
        if i == 0:
            # sum tp;sum fp
            final_type_nd[i, 3] = type_nd[sort_index[i], 1]
            final_type_nd[i, 4] = type_nd[sort_index[i], 2]
        else:
            final_type_nd[i, 3] = final_type_nd[i - 1, 3] + type_nd[sort_index[i], 1]
            final_type_nd[i, 4] = final_type_nd[i - 1, 4] + type_nd[sort_index[i], 2]
        # pre,recall
        final_type_nd[i, 5] = final_type_nd[i, 3] / (final_type_nd[i, 3] + final_type_nd[i, 4])
        final_type_nd[i, 6] = final_type_nd[i, 3] / type_all_gts

    precion_nd = final_type_nd[:, 5]
    recall_nd = final_type_nd[:, 6]
    return precion_nd,recall_nd

def get_fps_sen_nd(type_all_nd,valid_image_num,valid_all_gts):
    sort_index = np.argsort(-type_all_nd[:, 0])
    final_type_nd = np.zeros((len(sort_index), 7), dtype=np.float32)  # scores,tp,fp,sum tp,sum fp,fps,sen
    for i in range(len(sort_index)):
        # scores,tp,fp
        final_type_nd[i, :3] = type_all_nd[sort_index[i], :3]
        if i == 0:
            # sum tp;sum fp
            final_type_nd[i, 3] = type_all_nd[sort_index[i], 1]
            final_type_nd[i, 4] = type_all_nd[sort_index[i], 2]
        else:
            final_type_nd[i, 3] = final_type_nd[i - 1, 3] + type_all_nd[sort_index[i], 1]
            final_type_nd[i, 4] = final_type_nd[i - 1, 4] + type_all_nd[sort_index[i], 2]
        # fps,sen
        final_type_nd[i, 5] = final_type_nd[i, 4] / valid_image_num     #fp/img_num
        final_type_nd[i, 6] = final_type_nd[i, 3] / valid_all_gts

    fps = final_type_nd[:, 5]
    sen = final_type_nd[:, 6]
    return fps,sen

def plot_x_y(x,y,x_label,y_label,title,out_put,epoch):
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    plt.plot(x,y,color='b')
    #plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('{}'.format(x_label))
    plt.ylabel('{}'.format(y_label))
    plt.title('{}'.format(title))
    plt.savefig('{}/{}'.format(out_put,epoch))
    plt.close()

def index_number(li, defaultnumber):
    select = Decimal(str(defaultnumber)) - Decimal(str(li[0]))
    index = 0
    for i in range(1, len(li) - 1):
        select2 = Decimal(str(defaultnumber)) - Decimal(str(li[i]))
        if (abs(select) > abs(select2)):
            select = select2
            index = i
    return index

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def get_brain_number_people(fileDir):
    meningioma = 0  # 708 image; 82 people
    meningioma_people = 0
    glioma = 0  # 1426 image; 89 people
    glioma_people = 0
    pituitary_tumor = 0  # 930 image; 62 people
    pituitary_tumor_people = 0
    for people in os.listdir(fileDir):
        people_path = os.path.join(fileDir, people)
        if people[-1] == '1':
            meningioma += len(os.listdir(people_path))
            meningioma_people += 1
        elif people[-1] == '2':
            glioma += len(os.listdir(people_path))
            glioma_people += 1
        elif people[-1] == '3':
            pituitary_tumor += len(os.listdir(people_path))
            pituitary_tumor_people += 1
        else:
            assert False, 'class error'
    return meningioma, glioma, pituitary_tumor

def get_brain_number(fileDir):
    meningioma = 0  # 708 image; 82 people
    meningioma_people = 0
    glioma = 0  # 1426 image; 89 people
    glioma_people = 0
    pituitary_tumor = 0  # 930 image; 62 people
    pituitary_tumor_people = 0
    for people in os.listdir(fileDir):
        f = h5py.File(os.path.join(fileDir,people),'r')
        label = int(f['cjdata/label'][:][0, 0])
        if label == 1:
            meningioma_people += 1
        elif label == 2:
            glioma_people += 1
        elif label == 3:
            pituitary_tumor_people += 1
        else:
            assert False, 'class error'
    print(meningioma_people,glioma_people,pituitary_tumor_people)
    return meningioma_people, glioma_people, pituitary_tumor_people

def plot_iou_results(label,output,pil_img,output_dir,people_name,img_index):
    score_list = []
    iou_list = []

    iou_error = 0
    both_error = 0
    class_error = 0
    right_num = 0

    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    gt_boxes=label['boxes'].detach().cpu().numpy().tolist()[0]
    gt_label = label['labels'].detach().cpu().numpy()[0]

    #print(people_name,img_index,gt_boxes)
    print(gt_boxes)
    print(output['boxes'])
    CLASSES = [
        'meningioma','glioma','pituitary tumor'
    ]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes

    #GT绘制
    ax.add_patch(plt.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                               fill=False, color='red', linewidth=3))
    text = f'{CLASSES[gt_label - 1]}'
    ax.text(gt_xmin, gt_ymax+40, text, fontsize=30,
            bbox=dict(facecolor='red', alpha=0.5))

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()

    index = np.argsort(-scores)
    c = 'green'
    for n in range(scores.shape[0]):
        i = index[n]
        score = scores[i]
        xmin, ymin, xmax, ymax = boxes[i,:]
        label = labels[i]

        #calculate iou
        x1 = max(xmin, gt_xmin)
        y1 = max(ymin, gt_ymin)
        x2 = min(xmax, gt_xmax)
        y2 = min(ymax, gt_ymax)
        if x2 - x1 < 0 or y2 - y1 < 0:
            iou = 0
        else:
            overlap = (x2 - x1) * (y2 - y1)
            union = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) + (xmax - xmin) * (ymax - ymin) - overlap
            iou = overlap / union

        if iou > 0.5 and label == gt_label:#正确预测
            right_num += 1
        elif iou < 0.5 and label == gt_label:#交并比低
            c = 'yellow'
            iou_error += 1
        elif iou > 0.5 and label != gt_label:#类别判断错误
            c = 'cyan'
            class_error += 1
        else:#均错
            c = 'blue'
            both_error += 1

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[label-1]}: {score:0.3f}'
        #text = f'{CLASSES[label - 1]}: {score:0.2f},{iou:0.2f}'
        ax.text(xmin, ymin-20, text, fontsize=30,color='yellow',
                bbox=dict(facecolor=c, alpha=0.5))

        score_list.append(score)
        iou_list.append(iou)

    final_output_dir = os.path.join(output_dir,c)
    if not os.path.isdir(final_output_dir):
        os.makedirs(final_output_dir)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('{}/{}.png'.format(final_output_dir,img_index))
    plt.close()

    error =[right_num,iou_error,class_error,both_error]
    return score_list, iou_list, error

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def grid_gray_image(imgs, each_row: int):
    #改为[N,C,H,W]
    '''
    imgs shape: batch * size (e.g., 64x32x32, 64 is the number of the gray images, and (32, 32) is the size of each gray image)
    '''
    row_num = imgs.shape[-3]//each_row
    for i in range(row_num):
        img = imgs[i*each_row]
        img = (img - img.min()) / (img.max() - img.min())
        for j in range(1, each_row):
            tmp_img = imgs[i*each_row+j]
            tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
            img = np.hstack((img, tmp_img))
        if i == 0:
            ans = img
        else:
            ans = np.vstack((ans, img))
    return ans