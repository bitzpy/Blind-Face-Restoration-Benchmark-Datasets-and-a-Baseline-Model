

import os
import numpy as np
import cv2
import torch
from tqdm import tqdm, trange

from ssim import SSIM, MS_SSIM
from inception import InceptionV3
from fid import calculate_frechet_distance as cfd
from torch.nn.functional import adaptive_avg_pool2d

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
DEV = torch.device('cuda')
LOG10 = torch.log(torch.tensor([10.0],dtype=torch.float32, device=DEV))

totensor = lambda x: torch.from_numpy(x.transpose(2,0,1) / 255.)

def load_result_tensor(files):
    '''
        Input: directory containing testing images & gt
        Output: torch cuda tensors
    '''
    l = len(files)
    s = 512
    
    fake = torch.zeros((l,3,s,s), dtype=torch.float32, device=DEV)
    real = torch.zeros((l,3,s,s), dtype=torch.float32, device=DEV)
    for i in trange(l):
        try:
            pack = totensor(cv2.imread(files[i]))
            fake[i] = pack[:,:,s:s*2]
            real[i] = pack[:,:,2*s:3*s]
            # fake[i] = pack[:, :s]
            # real[i] = pack[:, s:]
        except:
            print('cyka blyat loading %s' % files[i])
            continue
        
    print('Loading Complete!\n', fake.shape, real.shape)
    return fake, real

def psnr(fake, real, eps=1e-8):
    EPS = torch.tensor([1e-8], dtype=fake.dtype, device=fake.device)
    res = fake - real
    mse = (res ** 2).mean(dim=(1,2,3))
    mse = torch.max(mse, EPS)  # numerical stability
    psnr_ = 10 * -torch.log(mse) / LOG10

    return psnr_.mean().item() 
    
def get_features(tensor, model, batch_size=50, dims=2048):
    model.eval()
    l = tensor.shape[0]
    if batch_size > l:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = l

    act = np.empty((l, dims))
    for i in trange(0, l, batch_size):
        batch = tensor[i: i+batch_size]
        pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act[i:i+batch_size] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    return act
    
def fid(fake, real, model):
    act_fake = get_features(fake, model)
    act_real = get_features(real, model)
    return act_fake, act_real
        
def run_main(FOLDER):
    file_packs = [os.path.join(FOLDER, l) for l in os.listdir(FOLDER)]
    
    print(len(file_packs))
    bs = 50
    file_chunks = [
        file_packs[i*bs:i*bs+bs] for i in range(len(file_packs) // bs)
    ]
    
    mean_psnr = 0
    mean_ssim = 0
    mean_msssim = 0
    act_fakes, act_reals = [], []
    for i, chunk in enumerate(file_chunks):
        print('Processing chunk %d' % i)
        fake, real = load_result_tensor(chunk)
        act_fake, act_real = fid(fake, real, fid_model)
        print(act_fake.shape)
        print(act_real.shape)
        act_fakes.append(act_fake)
        act_reals.append(act_real)
        
        mean_psnr += psnr(fake, real)
        mean_ssim += ssim(fake, real).mean().item()
        mean_msssim += msssim(fake, real).mean().item()
        print(mean_psnr)
        #print(mean_ssim)
        #print(mean_msssim)
    #print('111')
    act_fake = np.concatenate(act_fakes, axis=0)
    act_real = np.concatenate(act_reals, axis=0)
    mf = np.mean(act_fake, axis=0)
    sf = np.cov(act_fake, rowvar=False)
    mr = np.mean(act_real, axis=0)
    sr = np.cov(act_real, rowvar=False)
    
    mean_fid = cfd(mf, sf, mr, sr)
    
    print(FOLDER)
    print('FID: ', mean_fid)
    print('Mean PSNR: ', mean_psnr / len(file_chunks))
    print('Mean SSIM: ', mean_ssim / len(file_chunks))
    print('Mean MS_SSIM: ', mean_msssim / len(file_chunks))
        
    with open(FOLDER + '_metrics.txt', 'w') as f:
        f.write('FID: %.6f\n' % mean_fid)
        f.write('Mean PSNR: %.6f\n' % (mean_psnr / len(file_chunks)))
        f.write('Mean SSIM: %.6f\n' % (mean_ssim / len(file_chunks)))
        f.write('Mean MS_SSIM: %.6f\n' % (mean_msssim / len(file_chunks)))
    
if __name__ == '__main__':
    ssim = SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()
    msssim = MS_SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=True).cuda()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3([block_idx]).cuda()

    folders = [
        
        

        '/home/SwinIR/results/jpeg/4',
        '/home/SwinIR/results/blur/4',
        '/home/SwinIR/results/noise/4',
        '/home/SwinIR/results/lr/4',
        '/home/SwinIR/results/full/4'
        

    ]
    for folder in folders:
        run_main(folder)
    
    #run_main(debug)



