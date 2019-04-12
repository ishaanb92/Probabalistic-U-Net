## PyTorch implementation of a U-Net

For details about the U-Net please refer to : http://arxiv.org/abs/1505.04597

In addition to the model, we also provide a PyTorch Dataset wrapper (WIP) for the [CHAOS Liver MR dataset](https://chaos.grand-challenge.org/). 

Dependecies:
* PyTorch
* Numpy
* imageio
* Tensorflow (To use [tensorboardX](https://github.com/lanpa/tensorboardX) for viz)


This project is a WIP. The goal is to have a stable implementation of the [Probablistic U-Net](https://arxiv.org/abs/1806.05034)


#### Usage
TODO


#### Pending tasks:
* Segmentation metrics
* Track training v/s validation losses
* Additional data augmentation using [gryds](https://github.com/tueimage/gryds)
* Hyper-parameter tuning
* Implementing Prior and Posterior nets for Prob. U-Net

Code has been tested on Python 3.7.2
