# -*- coding: utf-8 -*-
# @Time : 2021/8/13 9:59
# @Author : Shen Junyong
# @File : brain_tumor_generate_coco
# @Project : code_for_classification_detection
import json
import numpy as np
import os
import time
import h5py

#对文件下脑数据集进行coco格式json文件生成
def generate_brain_img_json(fileDir,name, tofiledir):
    #fileDir:要生成json文件的图片上级文件夹名
    #name:文件保存名
    #tofiledir:保存json文件的路径
    info={
        "year": 2021, "version": 'the first version', "description": 'brain_tumor_dataset tumor', "contributor": 'shenjunyong', "url": 'None', "date_created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    image=[]
    license=[]
    annotation=[]
    categories=[]

    number_asscii = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
    letter_asscii = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                     90]
    letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

    for img_name in os.listdir(fileDir):
        img_file_path = os.path.join(fileDir, img_name)
        f = h5py.File(img_file_path, 'r')

        data = np.asarray(f['cjdata/PID']).tolist()
        image_id = int(img_name.replace('.mat', ''))
        annotation_id = 200000 + image_id
        people_id = ''
        for i in range(len(data)):
            try:
                people_id += str(number_asscii.index(data[i][0]))  # 改字符串形式！！数值存在问题
            except ValueError:
                people_id += letter[letter_asscii.index(data[i][0])]

        label = int(f['cjdata/label'][:][0, 0])
        w,h = f['cjdata/image'].shape
        pre_point_list = np.asarray(f['cjdata/tumorBorder']).tolist()[0]
        point_nd = np.zeros((0, 2), dtype=np.float64)
        for i in range(0, len(pre_point_list), 2):
            point_nd = np.append(point_nd, np.asarray([pre_point_list[i+1], pre_point_list[i]]).reshape(1, 2),
                                 axis=0)
        x_max = int(np.max(point_nd[:, 0]))
        x_min = int(np.min(point_nd[:, 0]))
        y_max = int(np.max(point_nd[:, 1]))
        y_min = int(np.min(point_nd[:, 1]))

        width = x_max - x_min
        height = y_max - y_min

        image_dict={"id": image_id,
                    "width": w,
                    "height": h,
                    "file_name": img_file_path,
                    "license": people_id,
                    "flickr_url": 'None',
                    "coco_url": 'None',
                    "date_captured": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    }
        license_dict={
            "id": people_id,
            "name": people_id,
            "url": 'None'
        }


        if label == 1:
            label_name = 'meningioma'
        elif label == 2:
            label_name = 'glioma'
        elif label == 3:
            label_name = 'pituitary tumor'
        else:
            assert False,'LABEL ERROR'

        annotation_dict = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": label,
            "segmentation": [[x_min, y_min, x_max, y_max]],
            "area": width * height,
            "bbox": [x_min, y_min, x_max, y_max],
            "iscrowd": 0
        }
        categories_dict={
            "id": label,
            "name": label_name,
            "supercategory": None
        }

        image.append(image_dict)
        annotation.append(annotation_dict)
        categories.append(categories_dict)
        license.append(license_dict)

    coco_label_dict={
        'info':info,
        'images':image,
        'annotations':annotation,
        'licenses':license,
        'categories':categories
    }

    data=json.dumps(coco_label_dict, ensure_ascii=False)
    with open(r'{}/{}.json'.format(tofiledir,name),'w') as f:
        f.write(data)

if __name__ == '__main__':
    tofiledir = "../data/brain_tvt/annotations"
    filedir = "../data/brain_tvt"
    generate_brain_img_json(r'{}/train'.format(filedir), name='train', tofiledir=tofiledir)
    generate_brain_img_json(r'{}/val'.format(filedir), name='val', tofiledir=tofiledir)
    generate_brain_img_json(r'{}/test'.format(filedir), name='test', tofiledir=tofiledir)