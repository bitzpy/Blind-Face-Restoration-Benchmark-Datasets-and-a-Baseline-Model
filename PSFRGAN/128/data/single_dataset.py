from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.utils import onehot_parse_map
class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        BaseDataset.__init__(self, opt)
        self.hr_size = 128
        self.hq_path = '/home/omnisky/storage/face/yyj/dataset/real_zpy'
        self.lq_path = '/home/omnisky/storage/face/yyj/dataset/real_zpy'
        self.mask_path = '/home/omnisky/storage/face/yyj/dataset/real_world/realworld_parse'
        # self.hq_path = '/home/face/yyj/dataset/celeba/celeba_gt128'
        # self.lq_path = '/home/face/yyj/dataset/celeba/celeba_x2full'
        # self.mask_path = '/home/face/yyj/dataset/celeba/fullparse'

        self.data = []
        self.data = os.listdir(self.hq_path)
        # for i in range (0, 500):
        #     if i == 234: continue
        #     print(i)
        #     tmp = os.listdir( self.hq_path +'/'+str(i))
        #     for j in range(len(tmp)):
        #         tmp[j] = str(i)+'/'+tmp[j]
        #     self.data = self.data+tmp
        print(len(self.data))
        #self.data  = os.listdir( self.hq_path )
        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        self.random_crop = transforms.RandomCrop(self.hr_size)
 
    def __getitem__(self, index):
        file = self.data[index]
        lq_path = os.path.join(self.lq_path, file)
        hq_path = os.path.join(self.hq_path, file)
        mask_path = os.path.join(self.mask_path, file)

        lr_img = Image.open(lq_path).convert('RGB')
        hr_img = Image.open(hq_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')
                    

      
        mask_img = mask_img.resize((self.hr_size, self.hr_size))
        mask_label = onehot_parse_map(mask_img)
        mask_label = torch.tensor(mask_label).float()

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        return {'LR': lr_tensor, 'LR_paths': file, 'mask':mask_label,  'gt_img': hr_tensor }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
