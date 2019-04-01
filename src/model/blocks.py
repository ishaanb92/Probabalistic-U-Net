"""
Class definitions for a standard U-Net Up-and Down-sampling blocks

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self,in_channels=1,filter_num,pool=True):
        super(EncoderBlock,self).__init__()
        self.filter_num = filter_num
        self.pool = pool
        if pool is True:
            self.max_pool=nn.MaxPool2d(kernel_size=2)
        self.conv_start = nn.Conv2d(in_channels=in_channels,out_channels=self.filter_num,kernel_size=3)
        self.conv_repeat = nn.Conv2d(in_channels=self.filter_num,out_channels=self.filter_num,kernel_size=3)

    def forward(self,x):
        if self.pool is True:
            x = self.max_pool(x)
        stage_1 = F.relu(self.conv_start(x))
        stage_2 = F.relu(self.conv_repeat(stage_1))
        stage_3 = F.relu(self.conv_repeat(stage_2))
        return stage_3


class DecoderBlock(nn.Module):
    def __init__(self,in_channels,skip_depth,filter_num,interpolate=False):
        # Up-sampling (interpolation or transposed conv) --> EncoderBlock
        super(DecoderBlock,self).__init__()
        self.filter_num = filter_num
        if interpolate:
            raise NotImplemented('Interpolation is still a TODO')
        else:
            #Depth is preserved during up-sampling in the original U-Net paper
            self.up_sample = nn.ConvTranspose2d(in_channels=in_channels,out_channels=in_channels,kernel_size=2)

        self.conv_block = EncoderBlock(in_channels=in_channels+skip_depth,filter_num=self.filter_num,pool=False)

    def forward(self,x,skip_layer):
        up_sample_out = F.relu(self.up_sample(x))
        merged_out = torch.cat([self.up_sample_out,skip_layer],1)
        out = F.relu(self.conv_block(merged_out))
        return out









