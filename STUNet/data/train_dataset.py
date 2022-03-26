import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import cv2
import torch

def onehot_parse_map(img):
    """
    input: RGB color parse map
    output: one hot encoding of parse map
    """
    n_label = len(MASK_COLORMAP)
    img = np.array(img, dtype=np.uint8)
    h, w = img.shape[:2]
    onehot_label = np.zeros((n_label, h, w))
    colormap = np.array(MASK_COLORMAP).reshape(n_label, 1, 1, 3)
    colormap = np.tile(colormap, (1, h, w, 1))
    for idx, color in enumerate(MASK_COLORMAP):
        tmp_label = colormap[idx] == img
        onehot_label[idx] = tmp_label[..., 0] * tmp_label[..., 1] * tmp_label[..., 2]
    return onehot_label

class train_dataset(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(train_dataset, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        self.paths_H = opt['dataroot_H']
        self.paths_L = opt['dataroot_L']
        self.data = []
        for i in range (1, 7):
            if(i == 4): continue
            print(i)
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
       
        H, W, _ = img_H.shape
        # --------------------------------
        # HWC to CHW, numpy(uint) to tensor
        # --------------------------------
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.)
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float().div(255.)



        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'name':file}

    def __len__(self):
        return self.dataset_size 