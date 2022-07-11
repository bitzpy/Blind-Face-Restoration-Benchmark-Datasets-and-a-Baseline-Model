'''
This is a simplified training code of GPEN. It achieves comparable performance as in the paper.

@Created by rosinality

@Modified by yangxy (yangtao9009@gmail.com)
'''
import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from data_loader.test_dataset import TestDataset
from face_model.gpen_model import FullGenerator, Discriminator

from loss.id_loss import IDLoss
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def sample_data(loader):
    while True:
        for batch in loader:
            yield batch



def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def test(args, loader, g_ema, device):
    loader = sample_data(loader)

    pbar = range(0, 145152)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)



    for idx in pbar:
        i = idx + args.start_iter

        if i > 145152:
            print('Done!')

            break

        degraded_img, real_img, name = next(loader)
        degraded_img = degraded_img.to(device)
        real_img = real_img.to(device)

        if get_rank() == 0:
            with torch.no_grad():
                g_ema.eval()
                sample, _ = g_ema(degraded_img)
                sample = torch.cat((degraded_img, sample, real_img), 3) 
                utils.save_image(
                    sample,
                    'real'+name[0],
                    nrow=args.batch,
                    normalize=True,
                    range=(-1, 1),
                )

          

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=False)
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--result', type=str, default='real')
    parser.add_argument('--model', type=str, default='ckpts/full/350000.pth')


    args = parser.parse_args()
    args.distributed = False
    os.makedirs(args.result, exist_ok=True)
    # for i in range(500):
    #     os.mkdir(args.result+'/'+str(i))
    device = torch.device('cuda')


    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device).to(device)
    
    tmp = torch.load(args.model, map_location=torch.device('cpu'))
    g_ema.load_state_dict(tmp['g_ema'])
    g_ema.to(device)
    g_ema.eval()
   

    
    
    dataset = TestDataset(args.path, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False, distributed=args.distributed),
        drop_last=True,
    )

    
    
    
    
    test(args, loader, g_ema, device)
   
