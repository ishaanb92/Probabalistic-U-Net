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

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr',type=float,help='Initial learning rate for optimization',default=3e-3)
    parser.add_argument('--data_dir',type=str,help='Directory where train and val data exist',default='/home/ishaan/probablistic_u_net/data')
    parser.add_argument('--batch_size',type=int,help='Training batch size',default=16)
    parser.add_argument('--epochs',type=int,help='Training epochs',default=10)
    parser.add_argument('--gpu_id',type=int,help='Supply the GPU ID (0,1 or 2 on saruman)',default=-1)
    parser.add_argument('--renew',action='store_true',help='If true, older checkpoints are deleted')
    parser.add_argument('--checkpoint_dir',type=str,help='Directory to save model parameters',default='/home/ishaan/probablistic_u_net/checkpoints')
    args = parser.parse_args()
    return args


def train(args):


    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')


    # Instance the Dataset and Dataloader classes
    tnfms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.ToTensor()])

    train_dataset = ChaosLiverMR(root_dir = args.data_dir,
                                 transforms=tnfms,
                                 renew=False)

    val_dataset = ChaosLiverMR(root_dir=args.data_dir,
                               transforms=None,
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

    #Instance the UNet model and optimizer
    model = UNet(image_size=256)
    optimizer = optim.Adam(model.parameters())

    #Delete/load old checkpoints
    if args.renew is True:
        try:
            shutil.rmtree(args.checkpoint_dir)
        except FileNotFoundError:
            pass

        os.makedirs(args.checkpoint_dir)
        epoch_saved = 0

    else:
        model,optimizer,epoch_saved= load_model(model=model,
                                                optimizer = optimizer,
                                                checkpoint_dir=args.checkpoint_dir,
                                                training=True)

    model.to(device)
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Start the training loop
    for epoch in range(args.epochs - epoch_saved):
        running_loss = 0.0
        for i,data in enumerate(train_dataloader):
            images,labels = data['image'].to(device), data['label'].to(device)

            optimizer.zero_grad() #Clear gradient buffers
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%10 == 0:
                print('[Epoch {} Iteration {}] Training loss : {}'.format(epoch,i,running_loss/10))
                running_loss = 0.0
        #Save model every epoch
        save_model(model=model,optimizer=optimizer,epoch=epoch,checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    args = build_parser()
    train(args)

