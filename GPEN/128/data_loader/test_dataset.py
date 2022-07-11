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


class TestDataset(Dataset):
    def __init__(self, path, resolution=128):
        label_paths = '/home/omnisky/storage/face/yyj/dataset/real_zpy'
        image_paths = '/home/omnisky/storage/face/yyj/dataset/real_zpy'
        self.data = os.listdir(label_paths)
        # for i in range (0,500):
        #     if i == 234: continue
        #     print(i)
        #     tmp = os.listdir(label_paths+'/'+str(i))
        #     for j in range(len(tmp)):
        #         tmp[j] = str(i)+'/'+tmp[j]
        #     self.data = self.data+tmp
        
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

        return img_lq, img_gt, file

