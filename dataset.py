import numpy as np
import cv2
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms

from glob import glob
from sklearn.model_selection import train_test_split
from utils import str2bool, count_params
import argparse


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.
        nplabel = np.empty((160, 160, 3))
        # nplabel = np.empty((240, 240, 3))
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))# (3,160,160)

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,nplabel




        #读图片（如jpg、png）的代码
        '''
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = color.gray2rgb(image)
        #image = image[:,:,np.newaxis]
        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))       
        return image, mask
        '''


if __name__ == '__main__':
    img_paths = glob('./data/trainImageDcm/*')
    mask_paths = glob('./data/trainMaskDcm/*')
    args = None
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)  # 加载验证集数据
    print(val_dataset)