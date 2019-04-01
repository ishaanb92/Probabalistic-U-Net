"""
A PyTorch Implementation of a U-Net
http://arxiv.org/abs/1505.04597

Author: Ishaan Bhat
i.r.bhat@umcutrecht.nl

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from blocks import *

class UNet(nn.Module):

    def __init__(image_dim=128,n_channels=1,base_filter_num=64,num_blocks=4,num_classes=2):
        super(UNet,self).__init__()
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()

        #Create the encoder path
        pool = False
        filter_num = base_filter_num
        in_channels = n_channels
        depths = []
        for block_id in range(num_blocks):
            down_block = EncoderBlock(in_channels=in_channels,filter_num=filter_num,pool=pool)
            pool = True
            #Save the depth, used to configure the DecoderBlock
            depths.append(filter_num)
            #Update conv parameters
            in_channels=filter_num
            filter_num = filter_num*2
            #Add to the list of layers
            self.contracting_path.append(down_block)

        #Create the decoder path
        for block_id in range(num_blocks):
            in_channels = filter_num
            skip_depth = depths[-1-block_id]
            up_block = DecoderBlock(in_channels=in_channels,skip_depth=skip_depth,filter_num=filter_num/2)
            #Update conv parameters
            in_channels = filter_num/2
            #Add to the list of layers
            self.expanding_path.append(up_block)

        #Create the output map
        self.output = nn.Conv2d(in_channels=base_filter_num,out_channels=num_classes,kernel_size=1)

    def forward(x):
        down_ops = []

        for stage,down_sample in enumerate(self.contracting_path):
            x = down_sample(x)
            if stage != self.num_blocks-1:
                down_ops.append(x)

        for stage,up_sample in enumerate(self.expanding_path):
            x = up_sample(x,down_ops[-stage-1])

        out = self.output(x)
        return out






