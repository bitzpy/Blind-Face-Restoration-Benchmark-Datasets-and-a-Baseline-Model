import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from utils import util_option as option
import torch
import cv2
from models.model_STUNet import ModelSTUNet
from data.train_dataset import train_dataset 
from PIL import Image
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def main(json_path='options/opt.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
 

    opt = option.parse(parser.parse_args().opt, is_train=True)
  

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        print(1)
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = train_dataset(dataset_opt)
            train_loader = DataLoader(train_set,
                                        batch_size=dataset_opt['dataloader_batch_size'],
                                        shuffle=dataset_opt['dataloader_shuffle'],
                                        num_workers=dataset_opt['dataloader_num_workers'],
                                        drop_last=True,
                                        pin_memory=True)

       

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = ModelSTUNet(opt)
    model.init_train()
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    current_step = 0
    for epoch in range(6):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1
        
            model.feed_data(train_data)

        
            model.optimize_parameters(current_step)

         
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                print(message)
            if current_step % opt['train']['checkpoint_save'] == 0 :
                print('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                save_path = opt['path']['models']
                save_path = save_path+'/'+'img'
                os.makedirs(save_path, exist_ok=True)
                images = model.current_visuals()
                
                E_img = tensor2uint(images['E'])
                H_img = tensor2uint(images['H'])
                L_img = tensor2uint(images['L'])
                out = np.concatenate((L_img,E_img,H_img), axis = 1)
                out = Image.fromarray(out)
                img_path = save_path + '/' + 'iter_{:8,d}.png'.format(current_step)
                print(img_path)
                out.save(img_path)
if __name__ == '__main__':
    main()