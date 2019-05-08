"""
Miscelleneous utility functions

"""
import torch
import imageio
import numpy as np
import os
import glob

def save_as_image(result_dir = None,image_batch=None,label_batch=None,preds_batch=None,fmt='png',prefix=None,n_channels=1,gpu_id=-1):
    """
    Take a batch of tensors (images, labels and predictions) and save the batch
    as a collection of image grids, each image grid being one image-label-prediction
    triplet from a single member of the batch.

    Parameters:
        result_dir (str or Path object) : Directory to store the images
        image_batch (torch.Tensor) : Image batch to be saved (batch_size x channels x height x width)
        label_batch (torch.Tensor) : Ground truth maps batch to be saved (batch_size x n_classes x height x width)
        preds_batch (torch.Tensor) : Model predictions batch to be saved (batch_size x n_classes x height x width)
        fmt (str) : Extension used to save the image
        prefix (str) : Used a prefix in the filename under which the image is saved

    Returns:
        None

    """

    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)

    #Convert torch tensors to numpy ndarray
    if gpu_id >= 0:
        image_batch = image_batch.cpu().numpy()
        label_batch = label_batch.cpu().numpy()
        preds_batch = preds_batch.cpu().numpy()
    else:
        image_batch = image_batch.detach().numpy()
        label_batch = label_batch.detach().numpy()
        preds_batch = preds_batch.detach().numpy()


    # Adjust dynamic range while converting to np.uint8
    # to avoid loss compression
    image_batch = adjust_dynamic_range(image=image_batch)
    label_batch = adjust_dynamic_range(image=label_batch)
    preds_batch = adjust_dynamic_range(image=preds_batch)

    # [C,H,W] -> [H,W,C]
    image_batch = image_batch.transpose((0,3,2,1))
    label_batch = label_batch.transpose((0,3,2,1))
    preds_batch = preds_batch.transpose((0,3,2,1))

    h,w = image_batch.shape[1],image_batch.shape[2]

    n_classes = label_batch.shape[3]

    for batch_idx in range(image_batch.shape[0]):
        image = image_batch[batch_idx,:,:,:]
        labels = label_batch[batch_idx,:,:,:]
        preds = preds_batch[batch_idx,:,:,:]

        #Init empty grid as np array
        image_grid = np.zeros((3*h,n_classes*w,n_channels),dtype=np.uint8)

        #Add image at the top
        image_grid[0:h,0:w,:] = image

        #Add grount truth and predicted maps
        for class_id in range(n_classes):
            image_grid[h:2*h,class_id*w:(class_id+1)*w,0] = labels[:,:,class_id]
            image_grid[2*h:3*h,class_id*w:(class_id+1)*w,0] = preds[:,:,class_id]

        #Save the image grid
        fname = os.path.join(result_dir,'{}_{}.{}'.format(prefix,batch_idx,fmt))

        imageio.imwrite(fname,image_grid)


def adjust_dynamic_range(image):
    """
    Images need to saved with uint8 intensity values.
    The pytorch model internall uses float32 or float64 representation,
    leading to lossy compression while saving the image

    Reference: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values

    Parameters:
        image (np.ndarray): A single image or batch of images

    Returns:
        adjusted_image (np.ndarray) : np.uint8 image with intensity values in range [0,255]


    """
    eps = 0.0001 #For numerical stability
    image = image.astype(np.float64)/(np.amax(image,axis=None) + eps)
    image = 255*image
    adjusted_image = image.astype(np.uint8)
    return adjusted_image


def convert_to_grayscale(image):
    """
    Convert DICOM image (uint16) to grayscale range (uint8)

    Parameters:
        image (numpy ndarray) : Numpy array uint16

    Returns:
        np.uint8 matrix that can be saved as a grayscale image

    """

    eps = 1e-5

    image  = np.array(image,dtype=np.float32)

    image = image/(np.amax(image) + eps) # [0,1] range

    image = np.multiply(image,255)

    return np.array(image,dtype=np.uint8)

def save_model(model=None,optimizer=None,epoch=None,checkpoint_dir=None):
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

    save_path = os.path.join(checkpoint_dir,'checkpoint_epoch_{}.pt'.format(epoch))

    torch.save(save_dict,save_path)


def load_model(model=None,optimizer=None,checkpoint_dir=None,training=False):
    """
    Function to load the PyTorch model

    Parameters:
        model (torch.nn.Module object) : Pytorch model whose parameters are to be loaded
        path (str or Path object) : Path to directory where the parameters are saved
        training (bool) : Set to true if training is to be resumed. Set to False if inference is to be performed

    Returns:
        model (torch.nn.Module): Model object with after loading the trainable parameters
        epoch (int) : Epoch where the model was frozen. None if training is false
        checkpoint(Python dictionary) : Dictionary that saves the state

    """
    checkpoint_file_paths = glob.glob(os.path.join(checkpoint_dir,'*.pt'))

    most_recent_checkpoint_file_path = select_last_checkpoint(checkpoint_file_paths)


    checkpoint = torch.load(most_recent_checkpoint_file_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if training is True:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        model.train()
    else:
        model.eval()

    return model,optimizer,epoch

def select_last_checkpoint(file_list):
    """
    Given a list of checkpoint files, selects the
    most recent checkpoint. The checkpoints are saved
    in the following format : PATH/checkpoint_epoch_<epoch_id>

    Parameters:
        file_list (Pyton list) : List of file paths to different checkpoint files

    Returns:
        last_checkpoint_file (str or Path object) : Path to the last saved checkpoint

    """
    epochs = [int(fname.split('/')[-1].split('.')[0].split('_')[-1]) for fname in file_list]
    index_latest_checkpoint = epochs.index(max(epochs))
    return file_list[index_latest_checkpoint]


def calculate_total_grad_norm(parameters):
    """
    Debug function, used to calculate the gradient norm
    during training to check effectiveness

    Parameters:
        parameters(List of torch.nn.parameter.Parameter objects) : Parameters of the model being analyzed

    Returns:
        grad_norm(float) : L2 norm of the gradients

    """
    with torch.no_grad():
        grads = []
        for p in parameters:
            if p.requires_grad is True:
                grads.append(p.grad.flatten().tolist())

        flattened_list=  [grad_value for layer in grads for grad_value in layer]
        grad_norm = np.linalg.norm(np.array(flattened_list,dtype=np.float32))
        return grad_norm

