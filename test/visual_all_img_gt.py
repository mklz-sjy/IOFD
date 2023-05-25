# -*- coding: utf-8 -*-
# @Time : 2022/1/5 19:54
# @Author : Shen Junyong
# @File : visual_all_img_gt
# @Project : faster_rcnn_pytorch
import os
import shutil
import h5py
from matplotlib import pyplot as plt
import numpy as np

def brain_mat_to_img(filedir,savefiledir):
    for img_name in os.listdir(filedir):
        img_path = os.path.join(filedir,img_name)
        f = h5py.File(img_path, 'r')
        img = np.asarray(f['cjdata/image'])
        pre_point_list = np.asarray(f['cjdata/tumorBorder']).tolist()[0]
        point_nd = np.zeros((0, 2), dtype=np.float64)
        for i in range(0, len(pre_point_list), 2):
            point_nd = np.append(point_nd, np.asarray([pre_point_list[i + 1], pre_point_list[i]]).reshape(1, 2),
                                 axis=0)
        gt_xmax = int(np.max(point_nd[:, 0]))
        gt_xmin = int(np.min(point_nd[:, 0]))
        gt_ymax = int(np.max(point_nd[:, 1]))
        gt_ymin = int(np.min(point_nd[:, 1]))
        label = int(f['cjdata/label'][:][0, 0])
        CLASSES = [
            'meningioma', 'glioma', 'pituitary tumor'
        ]

        plt.figure(figsize=(40, 40))
        img_name = img_name.replace('.mat','.png')
        plt.imshow(img)
        ax = plt.gca()


        ax.add_patch(plt.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                                   fill=False, color='red', linewidth=3))
        text = f'{CLASSES[label - 1]}'
        ax.text(gt_xmax, gt_ymax, text, fontsize=15,
                bbox=dict(facecolor='red', alpha=0.5))
        ax.add_patch(plt.Polygon((point_nd), fill=False, color='red', linewidth=3))

        plt.savefig(os.path.join(savefiledir,img_name))
        plt.close()


if __name__ == '__main__':
    brain_mat_to_img(r'../data/brain_tvt/test',r'../data/img_brain')