import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from argparse import ArgumentParser
import os,sys
sys.path.append(os.path.join(os.getcwd(),'src'))
sys.path.append(os.path.join(os.getcwd(),'src','model'))
from unet import UNet
from data import ChaosLiverMR
from utils import *
import shutil
from tensorboardX import SummaryWriter
from metrics import calculate_dice_similairity

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr',type=float,help='Initial learning rate for optimization',default=1e-4)
    parser.add_argument('--data_dir',type=str,help='Directory where train and val data exist',default='/home/ishaan/Work/unet/Probabalistic-U-Net/data')
    parser.add_argument('--batch_size',type=int,help='Training batch size',default=16)
    parser.add_argument('--epochs',type=int,help='Training epochs',default=10)
    parser.add_argument('--gpu_id',type=int,help='Supply the GPU ID (0: Titan Xp, 1: Quadro P1000)',default= 0)
    parser.add_argument('--renew',action='store_true',help='If true, older checkpoints are deleted')
    parser.add_argument('--checkpoint_dir',type=str,help='Directory to save model parameters',default='/home/ishaan/Work/unet/Probabalistic-U-Net/checkpoints')
    parser.add_argument('--log_dir',type=str,help='Directory to store tensorboard logs',default='./logs')
    parser.add_argument('--seed',type=int,help='Fix seed for reproducibility',default=42)
    args = parser.parse_args()
    return args


def train(args):

    torch.manual_seed(args.seed)

    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    # Instance the Dataset and Dataloader classes
    train_dataset = ChaosLiverMR(root_dir = args.data_dir,
                                 train=True,
                                 renew=args.renew)

    val_dataset = ChaosLiverMR(root_dir=args.data_dir,
                               train=False,
                               renew=False)


    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size,
                                  shuffle=True,
                                  num_workers = 4)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size = args.batch_size,
                                shuffle=True,
                                num_workers = 4)


    train_results_dir = os.path.join(args.log_dir,'visualizations','train_preds')
    val_results_dir = os.path.join(args.log_dir,'visualizations','val_preds')

    #Instance the UNet model and optimizer
    model = UNet(image_size=256,num_classes=4)
    optimizer = optim.Adam(model.parameters())

    #Delete/load old checkpoints
    if args.renew is True:
        try:
            shutil.rmtree(args.checkpoint_dir)
        except FileNotFoundError:
            pass

        try:
            shutil.rmtree(train_results_dir)
        except FileNotFoundError:
            pass

        try:
            shutil.rmtree(val_results_dir)
        except FileNotFoundError:
            pass

        os.makedirs(args.checkpoint_dir)
        os.makedirs(train_results_dir)
        os.makedirs(val_results_dir)
        epoch_saved = 0

    else:
        model,optimizer,epoch_saved= load_model(model=model,
                                                optimizer = optimizer,
                                                checkpoint_dir=args.checkpoint_dir,
                                                training=True)

    # The optimizer and model must reside on the same device.
    # optimizer.to(device) method does not exist, therefore for an optimizer loaded from disk, we need to manually copy it over.
    # See: https://github.com/pytorch/pytorch/issues/2830
    if args.gpu_id>=0:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    model.to(device)

    # Define the loss function
    # reduction set to 'none' for debug purposes. FIXME
    criterion = nn.CrossEntropyLoss()

    # Set up logging
    writer = SummaryWriter(os.path.join(args.log_dir,'exp_{}'.format(args.lr)))

    # Start the training loop
    running_loss = []
    for epoch in range(epoch_saved,args.epochs):
        for i,data in enumerate(train_dataloader):
            images,labels = data['image'].to(device).float(), data['label'].to(device).float()

            # nn.CrossEntropy() loss-term is meant for multi-class
            # classification. However, instead of a one-hot label,
            # it needs the "class-id" at each spatial location of
            # the segmentation map. However, the one-hot class-maps
            # are useful to visualize the reference segmentations
            targets = torch.argmax(labels,dim=1)

            optimizer.zero_grad() #Clear gradient buffers
            outputs = model(images)


            loss = criterion(outputs,targets)

            # Loss diagnostics START
            # Idea: The avg. loss at each spatial point of the output map
            # must be -(1/log(n_classes)) at init

            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            writer.add_scalar('Training Loss',loss.item(),len(train_dataloader)*epoch+i)
            if i%50 == 0:
                with torch.no_grad():
                    # Visualize results
                    save_as_image(result_dir=train_results_dir,
                                  image_batch=images,
                                  label_batch = labels,
                                  preds_batch = outputs,
                                  prefix = 'train_epoch_{}_iter_{}'.format(epoch,i),
                                  gpu_id = args.gpu_id)

                    mean_train_dice = calculate_dice_similairity(seg=outputs,gt=labels)

                    # Calculate validation loss and metrics
                    running_val_loss = []
                    val_dice_scores = []
                    for val_idx,val_data in enumerate(val_dataloader):
                        val_images,val_labels = val_data['image'].to(device).float(),val_data['label'].to(device).float()

                        val_targets = torch.argmax(val_labels,dim=1)

                        val_outputs = model(val_images)

                        val_loss = criterion(val_outputs,val_targets)
                        running_val_loss.append(val_loss.item())

                        writer.add_scalar('Validation Loss',val_loss.item(),len(val_dataloader)*epoch+val_idx)

                        save_as_image(result_dir=val_results_dir,
                                      image_batch=val_images,
                                      label_batch = val_labels,
                                      preds_batch = val_outputs,
                                      prefix = 'val_epoch_{}_iter_{}_idx_{}'.format(epoch,i,val_idx),
                                      gpu_id = args.gpu_id)


                        val_dice_scores.append(calculate_dice_similairity(seg=val_outputs,gt=val_labels))

                    mean_train_loss = np.mean(np.array(running_loss))
                    mean_val_loss = np.mean(np.array(running_val_loss))
                    mean_val_dice = np.mean(np.array(val_dice_scores),axis=None)

                    print('[Epoch {} Iteration {}] Training loss : {} Validation loss : {}'.format(epoch,i,mean_train_loss,mean_val_loss))
                    print('[Epoch {} Iteration {}] (Training) Mean Dice Metric : {}'.format(epoch,i,mean_train_dice))
                    print('[Epoch {} Iteration {}] (Validation) Mean Dice Metric : {}'.format(epoch,i,mean_val_dice))
                    print('\n')

                    running_loss = []

        #Save model every epoch
        save_model(model=model,optimizer=optimizer,epoch=epoch,checkpoint_dir=args.checkpoint_dir)

    #writer.close()

if __name__ == '__main__':
    args = build_parser()
    train(args)

