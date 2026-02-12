<p align="center">
    <!-- license badge -->
    <a href="https://github.com/aicodeclub/CDGaussian_Code/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <!-- stars badge -->
    <a href="https://github.com/aicodeclub/CDGaussian_Code/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/aicodeclub/CDGaussian_Code?style=social"/>
    </a>
    <a href="https://github.com/aicodeclub/CDGaussian_Code/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/aicodeclub/CDGaussian_Code"/>
    </a>

</p>


<div align="center">

CDGaussian
===========================
_<h4>Large-Scale 3D Gaussian Splatting Pruning via Collective Decision</h4>_

### [Project Page](https://aicodeclub.github.io/CDGaussian/)

<div align="left">

<div align="center">
    <img src="assets/teaser2.png" width="900">
</div>


# Overview

We propose **CDGaussian**, a novel pruning framework for large-scale 3DGS that integrates collective decision, merging similar
Gaussians, cross-GPU probability synchronization, PSNR-based evaluation feedback to address core challenges in
the field. Our framework achieves up to **9×** model-size compression during training while maintaining high rendering quality,
regardless of scene scale. Extensive experiments demonstrate that CDGaussian surpasses all existing methods, indicating strong potential for 3D reconstruction.

## Setup

### Cloning Code

```shell
git clone https://github.com/aicodeclub/CDGaussian_Code.git --recursive
```

### Pytorch Environment

```
conda env create --file environment.yml
conda activate cdgaussian
```

**Since conda downloads are very slow, please use mamba instead to speed up the installation process.**


## Training

Single GPU, non-distributed training:
```shell
python train.py -s <path to COLMAP dataset>
```

2 GPU, batchSize = 2,distributed training
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --bsz 2 -s <path to COLMAP dataset>
```

4 GPU, batchSize = 4, distributed training:
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s <path to COLMAP dataset>
```

**Please see train_images.py for detailed instructions.**


# License

See `LICENSE.txt` for more information.

# Acknowledgements
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting), 
[Grendel-GS](https://github.com/nyu-systems/Grendel-GS)
