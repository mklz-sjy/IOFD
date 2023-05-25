# -*- coding: utf-8 -*-
# @Time : 2021/12/22 17:20
# @Author : Shen Junyong
# @File : Brain_tumor
# @Project : faster_rcnn_pytorch
from pathlib import Path
import torch
import numpy as np
import h5py
from PIL import Image
import util.transforms as T

from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        f = h5py.File(path, 'r')
        img = np.array(f['cjdata/image'])
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img*255
        img = Image.fromarray(img).convert('L')#灰度图

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

class Brain_Tumor(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(Brain_Tumor, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._format=ConvertBrain_TumorPolysToMask()
    def __getitem__(self, idx):
        img, target = super(Brain_Tumor, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self._format(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        assert target['boxes'].numel(), 'target boxes is null'
        return img, target

class ConvertBrain_TumorPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]#[[box1],[box2],[box3]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

def make_Brain_Tumor_transforms(image_set):#参数相应调整
    #for image
    normalize = T.Compose([
        T.ToTensor(),#divided by 255
        #T.Normalize()#faster rcnn has used.
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomPad(3),
            T.ColorJitter('brightess'),
            T.ColorJitter('contrast'),
            T.ColorJitter('saturation'),
            T.ColorJitter('hue'),
            normalize,
            T.RandomErasing(),
        ])

    #valid and test
    if image_set == 'val' or image_set == 'test'or image_set == 'sample':
        return T.Compose([
            normalize
        ])
    raise ValueError(f'unknown {image_set}')

def build_brain_tumor(image_set, path):
    root = Path(path)
    assert root.exists(), f'provided Brain Tumor path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "val": (root / "val", root / "annotations" / 'val.json'),
        "test":(root / "test", root/"annotations" / 'test.json')
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = Brain_Tumor(img_folder, ann_file, transforms=make_Brain_Tumor_transforms(image_set))
    return dataset, img_folder, ann_file