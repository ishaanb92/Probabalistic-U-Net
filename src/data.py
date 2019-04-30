from __future__ import print_function, division
import os
import torch
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
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
import imageio
import random
import nrrd
from utils import convert_to_grayscale
from scipy.misc import imresize

NUM_TRAIN_PATIENTS=15

class ChaosLiverMR(Dataset):
    def __init__(self,root_dir='./data',mode='T2SPIR',transforms=None,keep_train_prob = 0.9,renew=True,train=True):
        self.data_dir = os.path.join(root_dir,'CHAOS_data')
        self.save_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.train = train

        #Init various directory paths
        self.train_dir = os.path.join(self.save_dir,'train_data')
        self.val_dir = os.path.join(self.save_dir,'val_data')

        self.train_images_dir = os.path.join(self.train_dir,'images')
        self.train_labels_dir = os.path.join(self.train_dir,'labels')

        self.val_images_dir = os.path.join(self.val_dir,'images')
        self.val_labels_dir = os.path.join(self.val_dir,'labels')

        if renew is True: #Create new train-val split
            self.create_train_val_sets()

        self.create_path_list()


    def create_train_val_sets(self,shuffle=True):
        """
        Split patients into training and validation sets

        """

        patient_ids = [patient_dir.split('_')[-1] for patient_dir in os.listdir(self.data_dir)]

        if shuffle is True:
            random.shuffle(patient_ids)

        train_patient_ids = patient_ids[:NUM_TRAIN_PATIENTS]
        val_patient_ids = patient_ids[NUM_TRAIN_PATIENTS:]

        self.create_image_label_slices(patient_ids=train_patient_ids,train=True)
        self.create_image_label_slices(patient_ids=val_patient_ids,train=False)

    def create_image_label_slices(self,patient_ids=[],train=True):
        """
        Saves image slices from the MRI along with
        corresponding segmentations class labels

        Parameters:
            patient_ids (Python list ): List of patient ids to create the image/label set
            train (bool)

        Returns:
            None

        """

        if train is True:
            out_dir = self.train_dir
        else:
            out_dir = self.val_dir

        img_dir = os.path.join(out_dir,'images')
        lab_dir = os.path.join(out_dir,'labels')

        try:
            shutil.rmtree(out_dir)
        except FileNotFoundError:
            pass

        os.makedirs(out_dir)
        os.makedirs(img_dir)
        os.makedirs(lab_dir)

        for example_num,idx in enumerate(patient_ids):
            patient_folder = os.path.join(self.data_dir,'Patient_{}'.format(idx))
            img_vol,_ = nrrd.read(os.path.join(patient_folder,'T2SPIR_image.nrrd'))
            lab_vol,_ = nrrd.read(os.path.join(patient_folder,'T2SPIR_mask.nrrd'))
            _,_,n_slices = img_vol.shape
            for slice_id in range(n_slices):
                image_slice = img_vol[:,:,slice_id]
                label_slice = lab_vol[:,:,slice_id]
                image_fname = 'img_{}_{}.png'.format(idx,slice_id)
                label_fname = 'lab_{}_{}.png'.format(idx,slice_id)
                imageio.imwrite(os.path.join(img_dir,image_fname),convert_to_grayscale(image_slice))
                imageio.imwrite(os.path.join(lab_dir,label_fname),label_slice)

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

        self.image_paths = glob.glob(os.path.join(data_dir,'images','*.png'))

        for path in self.image_paths:
            patient_idx = path.split('/')[-1].split('.')[0].split('_')[1]
            slice_idx = path.split('/')[-1].split('.')[0].split('_')[-1]
            self.label_paths.append(os.path.join(data_dir,'labels','lab_{}_{}.png'.format(patient_idx,slice_idx)))


    def create_binary_class_maps(self,label,num_classes=5):
        """
        Take a label image having pixel values from 0-4
        and creates 5 binary class maps

        """

        class_map = np.zeros((num_classes,label.shape[0],label.shape[1]),dtype=np.uint8)
        for class_id in range(num_classes):
            class_mask = np.where(label==class_id,1,0)
            class_map[class_id,:,:] = class_mask

        return class_map

    def __getitem__(self,index):

        img_path = self.image_paths[index]
        label_path = self.label_paths[index]


        # Fix dtype for PIL conversion
        img = imageio.imread(img_path)
        label = imageio.imread(label_path)

        # Reshape for PIL conversion
        # We need to convert the arrays into the PIL format
        # because PyTorch transforms like 'Resize' etc.
        # operate on PIL format arrays
        img = img.reshape((img.shape[0],img.shape[1],1))
        class_maps = self.create_binary_class_maps(label)

        sample = {'image':img,'label': None}

        num_classes = class_maps.shape[0]


        if self.transforms is not None:

            # Convert to PIL + Resize image
            sample['image'] = self.transforms(sample['image'])
            new_h,new_w = sample['image'].size

            # Init the dictionary entry for 'label' because now we know the shape
            sample['label'] = np.zeros((num_classes,new_h,new_w),dtype=np.uint8)

            for class_id in range(num_classes):
                #FIXME : Deprecation warning for imresize, fix this soon
                sample['label'][class_id] = np.array(imresize(class_maps[class_id,:,:],size=(new_h,new_w)),dtype=np.uint8)

            # Make sure that values in the label matrix along class axis (first dimenstion)
            # sum up to 1
            np.testing.assert_array_equal(x=np.array(np.sum(sample['label'],axis=0),dtype=np.uint8),
                                          y=np.ones((new_h,new_w),dtype=np.uint8),
                                          err_msg="Values in the label matrix do not sum up to 1 along the class axis",
                                          verbose=True)

            # Transform to PyTorch tensor
            sample['image'] = TF.to_tensor(sample['image'])

        return sample


    def __len__(self):
        return len(self.image_paths)



if __name__ == '__main__':
    # Basic sanity for the dataset class -- Run this when making any change to this code

    tnfms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256)])

    chaos_dataset = ChaosLiverMR(root_dir='/home/ishaan/probablistic_u_net/data',
                                 mode='T2SPIR',
                                 transforms=tnfms,
                                 train=True,
                                 renew=True)
    #DataLoader
    dataloader = DataLoader(dataset=chaos_dataset,
                            batch_size = 1,
                            shuffle=True,
                            num_workers = 1)

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

            print('Max pixel value in image : {}'.format(np.amax(img)))

            #PyTorch returns image/label matrices in the range 0-1 with np.float64 format (through some internal [and undocumented] magic!)
            imageio.imwrite(os.path.join(test_batch_dir,'img_{}_{}.jpg'.format(iters,batch_idx)),convert_to_grayscale(img[0,:,:]))

            for class_id in range(label.shape[0]):
                imageio.imwrite(os.path.join(test_batch_dir,'label_{}_{}_{}.jpg'.format(iters,batch_idx,class_id)),convert_to_grayscale(label[class_id,:,:]))

        iters += 1
        if iters == 5:
            break

