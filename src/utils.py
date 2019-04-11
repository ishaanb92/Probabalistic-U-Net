"""
Miscelleneous utility functions

"""
import torch
import imageio
import numpy as np
import os

def save_as_image(result_dir = None,batch=None,rescale=True,fmt='png',prefix=None):
    """
    Function save a batch of tensors as images

    Parameters:
        result_dir (str or Path object) : Directory to store the images
        batch (torch.Tensor) : Image batch to be saved (batch_size x channels x height x width)
        rescale (bool) : If true, pixel values scaled to 0-255 (uint8)
        fmt (str) : Extension used to save the image
        prefix (str) : Used a prefix in the filename under which the image is saved

    Returns:
        None

    """

    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)


    #Convert torch tensors to numpy ndarray
    batch = batch.numpy()

    if rescale is True:
        batch = np.array(np.multiply(batch,255),dtype=np.uint8)
    else:
        batch = np.array(batch,dtype=np.uint8)

    # [C,H,W] -> [H,W,C]
    batch = batch.transpose((3,2,1))

    for image_idx in range(batch.shape[0]):
        image = batch[image_idx,:,:,:]
        fname = os.path.join(result_dir,'{}_{}.{}'.format(prefix,image_idx,fmt))
        imageio.imwrite(fname,image)


def save_model(model=None,optimizer=None,epoch=None,save_dir=None):
    """
    Function save the PyTorch model along with optimizer state

    Parameters:
        model (torch.nn.Module object) : Pytorch model whose parameters are to be saved
        optimizer (torch.optim object) : Optimizer used to train the model
        epoch (int) : Epoch the model was saved in
        path (str or Path object) : Path to directory where model parameters will be saved

    Returns:
        None

    """

    if model is None:
        print('Save operation failed because model object is undefined')
        return

    if optimizer is None:
        print('Save operation failed because optimizer is undefined')
        return

    save_dict = {'epoch' : epoch,
                 'model_state_dict' : model.state_dict(),
                 'optimizer_state_dict' : optimizer.state_dict()
                }

    save_path = os.path.join(save_dir,'checkpoint_epoch_{}.pt'.format(epoch))

    torch.save(save_dict,save_path)


def load_model(model=None,optimizer = None,save_dir=None,training=False):
    """
    Function to load the PyTorch model

    Parameters:
        model (torch.nn.Module object) : Pytorch model whose parameters are to be loaded
        path (str or Path object) : Path to directory where the parameters are saved
        training (bool) : Set to true if training is to be resumed. Set to False if inference is to be performed

    Returns:
        model (torch.nn.Module): Model object with after loading the trainable parameters
        optimizer (torch.optim object) : Optimizer with hyper-parameters picked up from checkpoint. 'None' if training is False
        epoch (int) : Epoch where the model was frozen. None if training is false

    """
    save_path = glob.glob(os.path.join(save_dir,'*.pt'))
    #TODO: Handle case where multiple checkpoint (from different epochs) are present
    checkpoints = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if training is True:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        model.train()
    else:
        model.eval()

    return model,optimizer,epoch




