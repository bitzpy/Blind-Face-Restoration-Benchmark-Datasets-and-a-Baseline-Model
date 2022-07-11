import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

from data.image_folder import make_dataset

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from utils.utils import onehot_parse_map

class FFHQDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = 512
        self.lr_size = 512
        self.hr_size = 512
        self.shuffle = True if opt.isTrain else False 
        # self.hq_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'gt')))
        # self.lq_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'gt')))
        # self.mask_dataset = sorted(make_dataset(os.path.join(opt.dataroot,'gt')))
        self.hq_path = '../512dataset/hr/train'
        self.lq_path = '../512dataset/lr/train/X2_X4_X8_full'
        self.mask_path = '../512dataset/lr/train/psfr_parse/tmp/X2_X4_X8'
        self.data = []
        for i in range (1, 7):
            if i == 4: continue
            print(i)
            tmp = os.listdir( self.hq_path +'/'+str(i))
            for j in range(len(tmp)):
                tmp[j] = str(i)+'/'+tmp[j]
            self.data = self.data+tmp
        print(len(self.data))

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        self.random_crop = transforms.RandomCrop(self.hr_size)

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        lq_path = os.path.join(self.lq_path, file)
        hq_path = os.path.join(self.hq_path, file)
        mask_path = os.path.join(self.mask_path, file)

        lr_img = Image.open(lq_path).convert('RGB')
        hr_img = Image.open(hq_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')
                    
        #hr_img = hr_img.resize((self.hr_size, self.hr_size))
        #hr_img = random_gray(hr_img, p=0.3)
      
        mask_img = mask_img.resize((self.hr_size, self.hr_size))
        mask_label = onehot_parse_map(mask_img)
        mask_label = torch.tensor(mask_label).float()

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': hq_path, 'Mask': mask_label}


def complex_imgaug(x, org_size, scale_size):
    """input single RGB PIL Image instance"""
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug_seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.GaussianBlur((3, 15)),
                iaa.AverageBlur(k=(3, 15)),
                iaa.MedianBlur(k=(3, 15)),
                iaa.MotionBlur((5, 25))
            ])),
            iaa.Resize(scale_size, interpolation=ia.ALL),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5)),
            iaa.Sometimes(0.7, iaa.JpegCompression(compression=(10, 65))),
            iaa.Resize(org_size),
        ])
    
    aug_img = aug_seq(images=x)
    return aug_img[0]


def random_gray(x, p=0.5):
    """input single RGB PIL Image instance"""
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug = iaa.Sometimes(p, iaa.Grayscale(alpha=1.0)) 
    aug_img = aug(images=x)
    return aug_img[0]

