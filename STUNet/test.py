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
from models.network_swinir import SwinFaceRestoration as net
from data.test_dataset import test_dataset 
from PIL import Image
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='noise', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')

    parser.add_argument('--model_path', type=str,
                        default='check_points_new/noise/models/300000_G.pth')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   

    
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    save_dir = 'results'+'/'+args.task
  
    window_size = 8
    os.makedirs(save_dir, exist_ok=True)
       
    # for i in range(1,7):
    #     os.mkdir('results'+'/'+args.task+'/'+str(i))
    test_set = test_dataset()
    test_loader = DataLoader(test_set,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                drop_last=True,
                                pin_memory=True)
    
    for i, test_data in enumerate(test_loader):
        print(i)
        imgL = test_data['L'].to(device)
        mask = test_data['mask'].to(device)
        imgH = test_data['H'].to(device)
 
        with torch.no_grad():
            output = test(torch.cat((imgL,mask),dim=1), model)
            #output = test(imgL, model)
        out_dict = OrderedDict()
        out_dict['L'] = imgL.detach()[0].float().cpu()
        out_dict['E'] = output.detach()[0].float().cpu()
        out_dict['H'] = imgH.detach()[0].float().cpu()
            
        E_img = tensor2uint(out_dict['E'])
        H_img = tensor2uint(out_dict['H'])
        L_img = tensor2uint(out_dict['L'])
        out = np.concatenate((L_img,E_img,H_img), axis = 1)
        out = Image.fromarray(out)
        img_path = save_dir + '/' + test_data['name'][0]

        #cv2.imwrite(img_path, out)
        out.save(img_path)


def define_model(args):
    
    model = net(in_chans=3,
                img_size=512,
                window_size=8,
                embed_dim=48,
                num_heads=[1,2,4,8],
                mlp_ratio=2)
    param_key_g = 'params'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model



def test(img_lq, model):

    output = model(img_lq)
   

    return output

if __name__ == '__main__':
    main()