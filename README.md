# **Contents**

0. [Benchmark Datasets Download Link](#{Benchmark-Datasets-Download-Link})

1. [STUNet](#STUNet)
2. [Blind Face Restoration Methods](#Blind-Face-Restoration-Methods)
3. [Pre-trained Models](#Pre-trained-Models)
4. [Benchmarking Results](#Benchmarking-Results)

# **Benchmark Datasets Download Link**

- **BFRBD128**
  - **Google Drive**:
  - **Baidu Netdisk**:https://pan.baidu.com/s/1sFpJeuBb5dSP8GETmGfnow (w2bx)
- **BFRBD512**
  - **Google Drive**:
  - **Baidu Netdisk**: https://pan.baidu.com/s/1LUHCqapQC5ePH1GaM6DJ4Q (k3vv)

# **STUNet**
###  **Environment**
- Ubuntu
- Pytorch 1.9
- CUDA 11.3
- Python_packages: torchvision,  torch,  numpy,  opencv-python,  Pillow,  timm,  ipython
- System_packages: libgl1-mesa-glx, libglib2.0-0
### **Dataset Preparation**

Download BFRBD128 and BFRBD512 from [Benchmark Datasets Download Link](#Benchmark-Datasets-Download-Link) and put them in the appropriate directory according to your needs. The two datasets provide

### **Testing**

Download the [pre-trained models](#Pre-trained-Models) of STUNet and put them in `STUNet/check_points/`.  Next, modify the path of datasets and pre-trained models in `test.py` .  Last, use `python test.py` to generate results.

### Training

You can also train our STUNet by yourself. The training configurations is stored in `STUNet/options/opt.json`. Modify the configuration file according to your needs. Then use `python train.py` to train the STUNet.

### Evaluation

# **Blind Face Restoration Methods**
```
DFDNet: https://github.com/csxmli2016/DFDNet
HiFaceGAN: https://github.com/chaofengc/PSFRGAN
PSFRGAN: https://github.com/Lotayou/Face-Renovation
GFPGAN: https://github.com/TencentARC/GFPGAN
GPEN: https://github.com/yangxy/GPEN
```
# **Pre-trained Models**
```
DFDNet:
HiFaceGAN: 
PSFRGAN: 
GFPGAN: 
GPEN: 
STUNet
```
# **Benchmarm Results**
```
DFDNet: 
HiFaceGAN: 
PSFRGAN:
GFPGAN: 
GPEN: 
STUNet
```
