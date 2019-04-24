from __future__ import print_function, division
import os
import torch
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import glob
from collections import OrderedDict
from collections import Counter
import shutil
import pydicom as dcm
import imageio

class ChaosLiverMR(Dataset):
    def __init__(self,root_dir='./data',mode='T2SPIR',transforms=None,keep_train_prob = 0.9,renew=True,train=True):
        self.root_dir = root_dir
        self.mode = mode
        self.keep_train_prob = keep_train_prob
        self.transforms = transforms
        self.train = train

        # Intensity intervals for different classes:
        # 1. Liver
        # 2. Right Kidney
        # 3. Left Kidnet
        # 4. Spleen
        # Interval values taken from : https://github.com/jonasteuwen/chaos-challenge/blob/master/build_chaos_dataset.py
        self.class_intervals = [(55,70),(110,135),(175,200),(240,255)]

        #Init various directory paths
        self.train_dir = os.path.join(self.root_dir,'train_data')
        self.val_dir = os.path.join(self.root_dir,'val_data')

        self.train_images_dir = os.path.join(self.train_dir,'images')
        self.train_labels_dir = os.path.join(self.train_dir,'labels')

        self.val_images_dir = os.path.join(self.val_dir,'images')
        self.val_labels_dir = os.path.join(self.val_dir,'labels')

        if renew is True: #Create new train-val split
            self.create_train_val_sets()

        self.create_path_list()


    def create_train_val_sets(self):
        """
        Creates folders for training and validation data
        Splits the data into training and validation sets
        based on keep_train_prob fraction

        """

        if os.path.exists(self.train_dir) is True:
            shutil.rmtree(self.train_dir)

        if os.path.exists(self.val_dir) is True:
            shutil.rmtree(self.val_dir)

        os.makedirs(self.train_images_dir)
        os.makedirs(self.train_labels_dir)
        os.makedirs(self.val_images_dir)
        os.makedirs(self.val_labels_dir)

        data_dict = OrderedDict()

        # Split images into 'train' and 'val' sets
        image_dir_list = [os.path.join(f.path,self.mode.upper(),'DICOM_anon') for f in os.scandir(os.path.join(self.root_dir,'Train_Sets','MR')) if f.is_dir()]
        label_dir_list = [os.path.join(f.path,self.mode.upper(),'Ground') for f in os.scandir(os.path.join(self.root_dir,'Train_Sets','MR')) if f.is_dir()]

        fnames = []
        for i_dir,l_dir in zip(image_dir_list,label_dir_list):
            images = glob.glob(os.path.join(i_dir,'*.dcm'))
            for image in images:
                fname = image.split('/')[-1].split('.')[0]
                fnames.append(fname)
                label = glob.glob(os.path.join(l_dir,fname+'.*'))[0]
                data_dict[image] = label

        counts_dict = Counter(fnames)


        num_train = int(self.keep_train_prob*len(data_dict))

        for idx,image_path in enumerate(data_dict):

            label_path = data_dict[image_path]

            if idx <= num_train:
                img_dst = os.path.join(self.train_images_dir,'img_{}.dcm'.format(idx))
                label_dst = os.path.join(self.train_labels_dir,'lab_{}.png'.format(idx))
            else:
                img_dst = os.path.join(self.val_images_dir,'img_{}.dcm'.format(idx))
                label_dst = os.path.join(self.val_labels_dir,'lab_{}.png'.format(idx))

            shutil.copy2(image_path,img_dst)
            shutil.copy2(label_path,label_dst)

    def create_path_list(self):
        """
        Function to create list of image and label paths
        that can be accesed by a numerical index in the
        __getitem__() method

        """
        self.image_paths = []
        self.label_paths = []

        if self.train is True:
            data_dir = self.train_dir
        else:
            data_dir = self.val_dir

        if os.path.exists(data_dir) is False:
            print('Train/validation data directories do no exist. Please set the renew argument to True while instancing the class')
            sys.exit()

        self.image_paths = glob.glob(os.path.join(data_dir,'images','*.dcm'))

        for path in self.image_paths:
            image_idx = path.split('/')[-1].split('.')[0].split('_')[1] # This is why I love Python!
            self.label_paths.append(os.path.join(data_dir,'labels','lab_{}.png'.format(image_idx)))



    def __getitem__(self,index):

        img_path = self.image_paths[index]
        label_path = self.label_paths[index]


        # Fix dtype for PIL conversion
        img = np.array(dcm.dcmread(img_path).pixel_array,dtype=np.uint8)
        label = np.array(imageio.imread(label_path),dtype=np.uint8)

        # Reshape for PIL conversion
        # We need to convert the arrays into the PIL format
        # because PyTorch transforms like 'Resize' etc.
        # operate on PIL format arrays
        img = img.reshape((img.shape[0],img.shape[1],1))
        label = label.reshape((label.shape[0],label.shape[1],1))

        sample = {'image':img,'label':label}

        num_classes = len(self.class_intervals)


        if self.transforms is not None:

            # Convert to PIL + Resize image
            sample['image'] = self.transforms(sample['image'])

            # PIL + Resize + numpy() for label for further splitting into class maps
            transformed_label_gray = np.array(self.transforms(sample['label']),dtype=np.uint8)

            # Split the single grayscale image into per-class binary maps
            class_maps = np.zeros((num_classes,transformed_label_gray.shape[0],transformed_label_gray.shape[1]))
            for class_id,interval in enumerate(self.class_intervals):
                class_maps[class_id,:,:] = self.create_class_seg_map(label_gray=transformed_label_gray,intensity_interval = interval)

            # Transform to PyTorch tensor
            sample['image'] = TF.to_tensor(sample['image'])
            sample['label'] = class_maps

        return sample

    def __len__(self):
        return len(self.image_paths)

    def create_class_seg_map(self,label_gray,intensity_interval = []):
        """
        Convert a single channel grayscale image to a multi-channel (=num classes)
        binary image

        Parameters:
            label_gray (np.array) : Gray-scale label image
            intesity_interval (Python list or tuple) : Interval to threshold
        Return:
            class_seg_map (np.array) : Segmentation map for the class

        """
        cond = np.logical_and((label_gray[:,:] > intensity_interval[0]),(label_gray[:,:] <= intensity_interval[1]))
        class_seg_map = np.array(cond,dtype=np.uint8)
        return class_seg_map



if __name__ == '__main__':
    # Basic sanity for the dataset class -- Run this when making any change to this code

    tnfms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256)])

    chaos_dataset = ChaosLiverMR(root_dir='/home/ishaan/probablistic_u_net/data/',
                                 mode='T2SPIR',
                                 transforms=tnfms,
                                 train=True,
                                 renew=True)
    #DataLoader
    dataloader = DataLoader(dataset=chaos_dataset,
                            batch_size = 4,
                            shuffle=True,
                            num_workers = 4)

    iters = 0

    test_batch_dir = 'test_data_batching'
    if os.path.exists(test_batch_dir) is True:
        shutil.rmtree(test_batch_dir)
    os.makedirs(test_batch_dir)

    # Generate batches and save them to a folder for viewing
    for sampled_batch in dataloader:
        batch_imgs = sampled_batch['image'].numpy()
        batch_labels = sampled_batch['label'].numpy()
        print('Label batch shape : {}'.format(batch_labels.shape))
        print('Image batch shape : {}'.format(batch_imgs.shape))
        for batch_idx in range (batch_imgs.shape[0]):
            img = batch_imgs[batch_idx]
            label = batch_labels[batch_idx]
            print('Max pixel value in seg map : {}'.format(np.amax(label[0,:,:])))
            print('Max pixel value in background map : {}'.format(np.amax(label[1,:,:])))

            #PyTorch returns image/label matrices in the range 0-1 with np.float64 format (through some internal [and undocumented] magic!)

            #For display, re-scale image to 0-255 range
            img = np.array(255*np.transpose(img, (1,2,0)),dtype=np.uint8)
            label = np.array(255*np.transpose(label, (1,2,0)),dtype=np.uint8)

            imageio.imwrite(os.path.join(test_batch_dir,'img_{}_{}.jpg'.format(iters,batch_idx)),img)
            imageio.imwrite(os.path.join(test_batch_dir,'label_{}_{}_seg.jpg'.format(iters,batch_idx)),label[:,:,0])
            imageio.imwrite(os.path.join(test_batch_dir,'label_{}_{}_back.jpg'.format(iters,batch_idx)),label[:,:,1])

        iters += 1
        if iters == 5:
            break

