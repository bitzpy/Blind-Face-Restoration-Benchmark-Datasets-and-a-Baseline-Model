import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import cv2
import torch
class test_dataset(data.Dataset):
    '''

    '''

    def __init__(self):
        super(test_dataset, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
      
        self.n_channels = 3
        self.patch_size = 128

        self.paths_H = '../../512dataset/hr/test'
        self.paths_L = '../../512dataset/lr/test/noise'
        self.data = []
        for i in range (4, 5):
            tmp = os.listdir(self.paths_H+'/'+str(i))
            for j in range(len(tmp)):
                tmp[j] = str(i)+'/'+tmp[j]
            self.data = self.data+tmp
        size = len( self.data)
        print(size)
        self.dataset_size = size

    def __getitem__(self, index):
        file = self.data[index]
            # get H image
        # ------------------------------------
        H_path = os.path.join(self.paths_H, file)
        img_H = cv2.imread(H_path)
        img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
       
        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = os.path.join(self.paths_L, file)
        img_L = cv2.imread(L_path)
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
        # --------------------------------
        # HWC to CHW, numpy(uint) to tensor
        # --------------------------------
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.)
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float().div(255.)


        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'name':file}

    def __len__(self):
        return self.dataset_size 