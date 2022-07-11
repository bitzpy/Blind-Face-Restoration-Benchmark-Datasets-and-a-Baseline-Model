import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import time 
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 4 # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.load_pretrain_models()

    save_dir = 'results/real'
    os.makedirs(save_dir+'/cat', exist_ok=True)
   
    print('creating result directory', save_dir)
    #netP = model.netP
    netG = model.netG
    model.eval()
    max_size = 9999 
    os.makedirs(os.path.join(save_dir, 'sr'), exist_ok=True)
    for i, data in tqdm(enumerate(dataset), total=len(dataset)//opt.batch_size):
        inp = data['LR']
        gt_img = data['gt_img']
        parse_map_sm = data['mask']
        with torch.no_grad():
            # parse_map, _ = netP(inp)
            # parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            output_SR = netG(inp, parse_map_sm)
        img_path = data['LR_paths']     # get image paths
        for i in tqdm(range(len(img_path))):
            inp_img = utils.batch_tensor_to_img(inp)
            gt = utils.batch_tensor_to_img(gt_img)
            output_sr_img = utils.batch_tensor_to_img(output_SR)
            #ref_parse_img = utils.color_parse_map(parse_map_sm)
            # inp_128 = utils.color_parse_map(inp128)
            # gt = utils.color_parse_map(gt_img)

            # save_path = os.path.join(save_dir, 'lq', os.path.basename(img_path[i]))
            # os.makedirs(os.path.join(save_dir, 'lq'), exist_ok=True)
            # save_img = Image.fromarray(inp_img[i])
            # save_img.save(save_path)

            # save_path = os.path.join(save_dir, 'hq', os.path.basename(img_path[i]))
            # os.makedirs(os.path.join(save_dir, 'hq'), exist_ok=True)
            # save_img = Image.fromarray(output_sr_img[i])
            # save_img.save(save_path)

            # save_path = os.path.join(save_dir, 'parse', os.path.basename(img_path[i]))
            # os.makedirs(os.path.join(save_dir, 'parse'), exist_ok=True)
            # save_img = Image.fromarray(ref_parse_img[i])
            # save_img.save(save_path)


            save_path = os.path.join(save_dir, 'cat', img_path[i])
           # print(save_path)
            os.makedirs(os.path.join(save_dir, 'cat'), exist_ok=True)
            #output = output_sr_img[i]
          
            output = np.concatenate((inp_img[i], output_sr_img[i], gt[i]), axis=1)
            save_img = Image.fromarray(output)
            save_img.save(save_path)

        if i > max_size: break


       
 
