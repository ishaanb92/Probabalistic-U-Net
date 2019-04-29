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
    parser.add_argument('--data_dir',type=str,help='Directory where train and val data exist',default='/home/ishaan/probablistic_u_net/data')
    parser.add_argument('--batch_size',type=int,help='Training batch size',default=8)
    parser.add_argument('--epochs',type=int,help='Training epochs',default=10)
    parser.add_argument('--gpu_id',type=int,help='Supply the GPU ID (0,1 or 2 on saruman)',default=-1)
    parser.add_argument('--renew',action='store_true',help='If true, older checkpoints are deleted')
    parser.add_argument('--checkpoint_dir',type=str,help='Directory to save model parameters',default='/home/ishaan/probablistic_u_net/checkpoints')
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
    tnfms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(256)])

    train_dataset = ChaosLiverMR(root_dir = args.data_dir,
                                 transforms=tnfms,
                                 renew=False)

    val_dataset = ChaosLiverMR(root_dir=args.data_dir,
                               transforms=tnfms,
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
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Set up logging
    writer = SummaryWriter(os.path.join(args.log_dir,'exp_{}'.format(args.lr)))

    # Start the training loop
    for epoch in range(epoch_saved,args.epochs):
        running_loss = 0.0

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

            loss_matrix = np.array(loss.tolist())

            # Calculate avg. loss value at each spatial point i.e. mean along the 0th axis
            spatial_loss_map = np.mean(loss_matrix,axis=0)

            # Loss diagnostics END

            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Log the loss
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

                    train_per_class_dice = []
                    for class_id in range(labels.shape[1]):
                        train_per_class_dice.append(calculate_dice_similairity(seg=outputs[:,class_id,:,:],gt=labels[:,class_id,:,:]))

                    # Calculate validation loss and metrics
                    val_loss = []
                    val_dice_scores = []
                    for val_idx,val_data in enumerate(val_dataloader):
                        val_images,val_labels = val_data['image'].to(device).float(),val_data['label'].to(device).float()

                        val_targets = torch.argmax(val_labels,dim=1)

                        val_outputs = model(val_images)
                        val_loss.append(torch.mean(criterion(val_outputs,val_targets)).item())

                        save_as_image(result_dir=val_results_dir,
                                      image_batch=val_images,
                                      label_batch = val_labels,
                                      preds_batch = val_outputs,
                                      prefix = 'val_epoch_{}_iter_{}_idx_{}'.format(epoch,i,val_idx),
                                      gpu_id = args.gpu_id)


                        per_class_val_dice_score = []

                        for class_id in range(val_labels.shape[1]):
                            per_class_val_dice_score.append(calculate_dice_similairity(seg=val_outputs[:,class_id,:,:],gt=val_labels[:,class_id,:,:]))

                        val_dice_scores.append(np.array(per_class_val_dice_score))

                    mean_train_loss = running_loss/50
                    mean_val_loss = np.mean(np.array(val_loss))

                    mean_train_dice = np.mean(np.array(train_per_class_dice))
                    mean_val_per_class_dice = np.mean(np.array(val_dice_scores),axis=0)
                    mean_val_dice = np.mean(np.array(val_dice_scores),axis=None)

                    print('[Epoch {} Iteration {}] Training loss : {} Validation loss : {}'.format(epoch,i,mean_train_loss,mean_val_loss))
                    print('[Epoch {} Iteration {}] (Training) Mean Dice Metric : {} Per-class dice metric : {}'.format(epoch,i,mean_train_dice,train_per_class_dice))
                    print('[Epoch {} Iteration {}] (Validation) Mean Dice Metric : {} Per-class dice metric : {}'.format(epoch,i,mean_val_dice,mean_val_per_class_dice))
                    print('\n')

                    running_loss = 0.0

        #Save model every epoch
        save_model(model=model,optimizer=optimizer,epoch=epoch,checkpoint_dir=args.checkpoint_dir)

    writer.close()

if __name__ == '__main__':
    args = build_parser()
    train(args)

