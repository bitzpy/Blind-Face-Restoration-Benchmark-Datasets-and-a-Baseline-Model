# **Contents**


1. [STUNet](#STUNet)
2. [Blind Face Restoration Methods](#Blind-Face-Restoration-Methods)
3. [Pre-trained Models](#Pre-trained-Models)
4. [Benchmarking Results](#Benchmarking-Results)



# **STUNet**
###  **Environment**
- Ubuntu
- Pytorch 1.9
- CUDA 11.3
- Python_packages: torchvision,  torch,  numpy,  opencv-python,  Pillow,  timm,  ipython
- System_packages: libgl1-mesa-glx, libglib2.0-0
### **Dataset Preparation**

Download [BFR128](https://github.com/HDCVLab/EDFace-Celeb-1M#edface-celeb-1m-bfr128--blind-face-restoration-hq-lq-blur-jpeg-artifact-noise-sr-full-full_x2-full_x4-full_x8) and [BFR512](https://github.com/HDCVLab/EDFace-Celeb-1M#edface-celeb-150k-bfr512-blind-face-restoration-hq-lq-blur-jpeg-artifact-noise-sr-full-full_x2-full_x4-full_x8), and put them in the appropriate directory according to your needs. The two datasets provide

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
DFDNet: https://github.com/csxmli2016/DFDNet (We use the pre-train model provided by the author.)
HiFaceGAN: https://pan.baidu.com/s/1Yiof155wF1GUNOuw2wUWXw (4kya)
PSFRGAN: https://pan.baidu.com/s/1GgTOc1FMF34b1Nf0_9rlGg (4kya)
GFPGAN: https://pan.baidu.com/s/1E7yt_FLLZghMJFYO8cSj9g (4kya)
GPEN: https://pan.baidu.com/s/1mySckCwIKIUGJbyhW28Idw (4kya)
STUNet: https://pan.baidu.com/s/16E4cqM7pbnh9w236l7_79Q (4kya)
```
# **Benchmark Results**
```
DFDNet: https://pan.baidu.com/s/1I2UysJ5vDfwGMvRgVVSqzg (exb4)
HiFaceGAN: https://pan.baidu.com/s/17tTcEYdy6AGNh9UZCQ4CiA (exb4)
PSFRGAN: https://pan.baidu.com/s/1fzcnJote018v6g3Tn1-URw (exb4)
GFPGAN: https://pan.baidu.com/s/1_P8OMwGaDyCe7H0G2a9fjA (exb4)
GPEN: https://pan.baidu.com/s/19OaTwlqvJlgOc_sIFESxpw (exb4)
STUNet: https://pan.baidu.com/s/1Mi-TlCmnFXgvY_jk17Tzeg (exb4) 

```
# **Acknowledgement**
Our codes are heavily based on [SwinIR](https://github.com/JingyunLiang/SwinIR). We also borrow some codes from [HiFaceGAN](https://github.com/Lotayou/Face-Renovation) and 
[Restormer](https://github.com/swz30/Restormer).


# **Citation**
If you think this work is useful for your research, please cite the following papers.

```
@inproceedings{zhang2022blind,
  title={Blind Face Restoration: Benchmark Datasets and a Baseline Model},
  author={Zhang, Puyang and Zhang, Kaihao and Luo, Wenhan and Li, Changsheng and Wang, Guoren},
  booktitle={arXiv:2206.03697},
  year={2022}
}

@inproceedings{zhang2022edface,
  title={EDFace-Celeb-1M: Benchmarking Face Hallucination with a Million-scale Dataset},
  author={Zhang, Kaihao and Li, Dongxu and Luo, Wenhan and Liu, Jingyu and Deng, Jiankang and Liu, Wei and Stefanos Zafeiriou},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2022}
}
```


