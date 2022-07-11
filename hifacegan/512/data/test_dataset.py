"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os, torch
from data.image_folder import make_dataset
import glob,os
import cv2
from random import randint

class MyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        self.data = []
        label_paths = '/home/omnisky/storage/face/yyj/dataset/real_zpy_512'
        image_paths = '/home/omnisky/storage/face/yyj/dataset/real_zpy_512'
        
        self.data = os.listdir(label_paths)
        # label_paths = label_paths[:opt.max_dataset_size]
        # image_paths = image_paths[:opt.max_dataset_size]
        #instance_paths = instance_paths[:opt.max_dataset_size]

        self.label_paths = label_paths
        self.image_paths = image_paths
        #self.instance_paths = instance_paths

        size = len( self.data)
        print(size)
        self.dataset_size = size

    def __getitem__(self, index):
        # Label Image
        file = self.data[index]
        #print(file)
        hq_path = os.path.join(self.label_paths, file)
        lq_path = os.path.join(self.image_paths, file)
        #print(lq_path)
        label = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(cv2.imread(hq_path), cv2.COLOR_BGR2RGB)                #lable和image是反的
                                                                                    #lable和image是反的
        label = label.transpose(2, 0, 1) / 255.                                     #lable和image是反的
        label = torch.from_numpy(label).to(torch.float32)

        image = image.transpose(2, 0, 1) / 255.
        image = torch.from_numpy(image).to(torch.float32)


        input_dict = {'label': label,
                      'instance': 0,
                      'image': image,
                      'path': file,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size




class TestDataset(MyDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = MyDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 512 if is_train else 512
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
