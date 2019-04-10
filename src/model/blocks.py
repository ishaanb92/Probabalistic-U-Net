"""
Class definitions for a standard U-Net Up-and Down-sampling blocks
http://arxiv.org/abs/1505.04597

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self,filter_num=64,in_channels=1):
        """
        Instances the Encoder block that forms a part of a U-Net
        Parameters:
            in_channels (int): Depth (or number of channels) of the tensor that the block acts on
            filter_num (int) : Number of filters used in the convolution ops inside the block, depth of the output of the enc block
            pool (bool) : Decides if a pooling layer is inserted at the start of the block (usually set to False for the first block in the UNet)

        Returns:
            out (torch.Tensor) : Output of operations performed in the block

        """
        super(EncoderBlock,self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,out_channels=self.filter_num,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=self.filter_num,out_channels=self.filter_num,kernel_size=3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        out = F.relu(self.conv2(x))
        return out


class DecoderBlock(nn.Module):
    def __init__(self,in_channels,concat_layer_depth,filter_num,interpolate=False):
        """
        Decoder block used in the U-Net

        Parameters:
            in_channels (int) : Number of channels of the incoming tensor for the upsampling op
            concat_layer_depth (int) : Number of channels to be concatenated via skip connections
            filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
            interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution

        Returns:
            out (torch.Tensor) : Output of the Decoder block
        """
        # Up-sampling (interpolation or transposed conv) --> EncoderBlock
        super(DecoderBlock,self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)

        if interpolate:
            raise NotImplemented('Interpolation is still a TODO')
        else:
            #Depth is preserved during up-sampling in the original U-Net paper
            self.up_sample = nn.ConvTranspose2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=2)


        self.conv_block = EncoderBlock(in_channels=self.in_channels+self.concat_layer_depth,filter_num=self.filter_num)

    def forward(self,x,skip_layer):
        up_sample_out = F.relu(self.up_sample(x))
        padded_up_sample_layer = self.pad_before_merge(up_sample_out,skip_layer)
        merged_out = torch.cat([padded_up_sample_layer,skip_layer],1)
        out = F.relu(self.conv_block(merged_out))
        return out

    def pad_before_merge(self,up_sample_layer,skip_layer):
        """
        Pads the up sampled layer to match the dims (H,W)
        of the skip layer before concatenation
        Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        Parameters:
            up_sample_layer (torch.Tensor) : Tensor holding the upsampled layer
            skip_layer (torch.Tensor) : Tensor holding the skip layer that needs to concatenated

        Returns:
            padded_up_sample_layer (torch.Tensor) : Padded tensor that can be merged



        """
        diffY = skip_layer.size()[2] - up_sample_layer.size()[2]
        diffX = skip_layer.size()[3] - up_sample_layer.size()[3]
        padded_up_sample_layer = F.pad(up_sample_layer, (diffX // 2, diffX - diffX//2,diffY // 2, diffY - diffY//2))
        return padded_up_sample_layer










