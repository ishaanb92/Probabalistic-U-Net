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


    # Rationale behind the conversion to numpy (and the evenetual migration of computation to the CPU):
    #   - Binarizing the tensor by applying the conditional casts it to a np ndarray of bool type
    #   - This np ndarray can't be cast back to a PyTorch tensor
    #   - Cleaner to convert the torch.Tensor to numpy.ndarray at the start
    #   - Through the type-check, we retain the flexibility to supply either a numpy array or a torch.Tensor as the input arguments

    if torch.is_tensor(seg) is True:
        seg = seg.detach()
        seg = seg.cpu().numpy()

    if torch.is_tensor(gt) is True:
        gt = gt.detach()
        gt = gt.cpu().numpy()

    #Binarize the images
    seg = (seg > 0.5).astype(int)
    gt = (gt > 0.5).astype(int)

    inter = np.inner(seg,gt)
    union = np.sum(seg) + np.sum(gt)

    eps = 0.0001 #For numerical stability

    dice_similarity_coeff = (2*inter.astype(float) + eps)/(union.astype(float) + eps)

    return dice_similarity_coeff






