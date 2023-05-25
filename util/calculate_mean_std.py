# -*- coding: utf-8 -*-
# @Time : 2021/12/22 17:42
# @Author : Shen Junyong
# @File : calculate_mean_std
# @Project : faster_rcnn_pytorch
import os
import h5py
import numpy as np
from PIL import Image

def get_brain_dataset_mean_std(args):
    train_folder = os.path.join(args.brain_tumor_path,'train')
    mean = 0
    std = 0
    num = 0
    for i in os.listdir(train_folder):
        num+=1
        img_path = os.path.join(train_folder,i)
        f = h5py.File(img_path, 'r')
        img = np.array(f['cjdata/image'])
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        mean+=img.mean()
        std+=img.std()
    mean = round(mean / num, 3)
    std = round(std / num, 3)
    print(mean)#0.158
    print(std)#0.161
    return mean,std

def get_AMD_dataset_mean_std(args):
    train_folder = os.path.join(args.AMD_PATH,'train')
    mean = 0
    std = 0
    num = 0
    for people in os.listdir(train_folder):
        people_path = os.path.join(train_folder,people)
        imgs_list=os.listdir(people_path)
        imgs_list.remove('Label')
        imgs_list.remove('Result')
        for img in imgs_list:
            num+=1
            image=Image.open(os.path.join(people_path,img)).convert('L')
            image_nd = np.array(image)
            image_nd = image_nd.astype(np.float32) / 255
            mean+=image_nd.mean()
            std+=image_nd.std()
    mean = round(mean / num, 3)
    std = round(std / num, 3)
    print(mean)
    print(std)
    return mean,std
