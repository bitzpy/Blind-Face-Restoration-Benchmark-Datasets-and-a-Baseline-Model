import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import degradations


class GFPGAN_degradation(object):
    def __init__(self):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 20]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20/255.
    
    def degrade_process(self, img_gt):
        if random.random() > 0.5:
            img_gt = cv2.flip(img_gt, 1)

        h, w = img_gt.shape[:2]
       
        # random color jitter 
        if np.random.uniform() < self.color_jitter_prob:
            jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.gray_prob:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_gt, img_lq

class FaceDataset(Dataset):
    def __init__(self, path, resolution=128):
        label_paths = '../512dataset/hr/train'
        image_paths = '../512dataset/lr/train/noise'
        self.data = []
        for i in range (1, 7):
            print(i)
            if i == 4 : continue
            tmp = os.listdir(label_paths+'/'+str(i))
            for j in range(len(tmp)):
                tmp[j] = str(i)+'/'+tmp[j]
            self.data = self.data+tmp
        
        # label_paths = label_paths[:opt.max_dataset_size]
        # image_paths = image_paths[:opt.max_dataset_size]
        #instance_paths = instance_paths[:opt.max_dataset_size]

        self.label_paths = label_paths
        self.image_paths = image_paths
        #self.instance_paths = instance_paths

        size = len( self.data)
        print(size)
        self.dataset_size = size
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        file = self.data[index]
        hq_path = os.path.join(self.label_paths, file)
        lq_path = os.path.join(self.image_paths, file)
   
        # img_lq = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)
        # img_gt = cv2.cvtColor(cv2.imread(hq_path), cv2.COLOR_BGR2RGB)               
                                                                                  
        # img_lq = img_lq.transpose(2, 0, 1) / 255.                                    
        # img_lq = torch.from_numpy(img_lq).to(torch.float32)

        # img_gt = img_gt.transpose(2, 0, 1) / 255.
        # img_gt = torch.from_numpy(img_gt).to(torch.float32)
        
        
        img_gt = cv2.imread(hq_path, cv2.IMREAD_COLOR)
        img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
    
        img_gt = img_gt.astype(np.float32)/255.
        img_lq = img_lq.astype(np.float32)/255.
        img_gt =  (torch.from_numpy(img_gt) - 0.5) / 0.5
        img_lq =  (torch.from_numpy(img_lq) - 0.5) / 0.5
        
        img_gt = img_gt.permute(2, 0, 1).flip(0) # BGR->RGB
        img_lq = img_lq.permute(2, 0, 1).flip(0) # BGR->RGB

        return img_lq, img_gt

