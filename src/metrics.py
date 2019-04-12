"""
Functions to calculate metrics

"""

import numpy as np
import torch

def calculate_dice_similairity(seg,gt):
    """
    Function that calculates the dice similarity co-efficient

    Parameters:
        seg (numpy nd array or torch.Tensor) : (Predicted )Segmentation map
        gt (numpy nd array or torch.Tensor) : Ground Truth

    Returns:
        dice_similarity_coeff (float) : Dice similiarty between predicted segmentations and ground truths

    """
    #Flatten
    seg = seg.view(-1)
    gt = gt.view(-1)

    if torch.is_tensor(seg) is True:
        seg = seg.detach()
        seg = seg.cpu().numpy()

    if torch.is_tensor(gt) is True:
        gt = gt.detach()
        gt = gt.cpu().numpy()

    #Binarize the images
    seg = (seg > 0.5).astype(float)
    gt = (gt > 0.5).astype(float)

    inter = np.inner(seg,gt)
    union = np.sum(seg) + np.sum(gt)

    eps = 0.0001 #For numerical stability

    dice_similarity_coeff = (2*inter.astype(float))/(union.astype(float) + eps)

    return dice_similarity_coeff






