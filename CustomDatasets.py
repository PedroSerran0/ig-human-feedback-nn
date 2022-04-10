# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:52:36 2022

@author: Fnac Home
"""
# Imports
import numpy as np
import os
from PIL import Image
from numpy import genfromtxt
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#=====================================================================================================
#======================================= ISIC 2017 ===================================================
#=====================================================================================================

# main_dir is the train/val/test directory, img_dir is inside each of those directories
def ISIC_map_images_and_labels(data_dir, label_file):
    
    # Get image_id and corresponding label from csv file
    labels = genfromtxt(os.path.join(data_dir, label_file), delimiter=',',encoding="utf8", dtype=None)
    labels = np.delete(labels,2,1)
    labels = np.delete(labels,0,0)
    
    # Images
    dir_files = os.listdir(data_dir)
    dir_imgs = [i for i in dir_files if i.split('.')[1]=='jpg']
    dir_imgs.sort()
    
    _labels_unique = np.unique(labels[:, 1])
    
    # Nr of Classes
    nr_classes = len(_labels_unique)

    # Create labels dictionary
    labels_dict = dict()
        
    for idx, _label in enumerate(_labels_unique):
        labels_dict[_label] = idx

    # Create img file name - image label array
    imgs_labels = np.column_stack((dir_imgs, labels[:,1]))
    
    return imgs_labels, labels_dict, nr_classes
    
# Create a Dataset Class
class ISIC17_Dataset(Dataset):
    def __init__(self, base_data_path, label_file, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.label_file = label_file
        self.base_data_path = base_data_path
        imgs_labels, self.labels_dict, self.nr_classes = ISIC_map_images_and_labels(base_data_path, label_file)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        
        # Open image with PIL
        image = Image.open(os.path.join(self.base_data_path, img_name))
        #plt.imshow(image)
        
        # Perform transformations with Numpy Array
        image = np.asarray(image)
        
        #image = np.reshape(image, newshape=(image.shape[0], image.shape[1], 1))
        #image = np.concatenate((image, image, image), axis=2)

        # Load image with PIL
        image = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label

#=====================================================================================================
#======================================= Aptos 2019 ==================================================
#=====================================================================================================

# main_dir is the train/val/test directory, img_dir is inside each of those directories
def Aptos19_map_images_and_labels(data_dir, label_file):
    
    # Get image_id and corresponding label from csv file
    labels = genfromtxt(os.path.join(data_dir, label_file), delimiter=',',encoding="utf8", dtype=None)
    labels = np.delete(labels,0,0)
    
    # Images
    dir_files = os.listdir(data_dir)
    dir_imgs = [i for i in dir_files if i.split('.')[1]=='png']
    dir_imgs.sort()
    
    _labels_unique = np.unique(labels[:, 1])
    # Nr of Classes
    nr_classes = len(_labels_unique)

    # Create labels dictionary
    labels_dict = dict()
        
    for idx, _label in enumerate(_labels_unique):
        labels_dict[_label] = idx

    # Create img file name - image label array
    imgs_labels = np.column_stack((dir_imgs, labels[:,1]))
    
    return imgs_labels, labels_dict, nr_classes


# Create a Dataset Class
class Aptos19_Dataset(Dataset):
    def __init__(self, base_data_path, label_file, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.label_file = label_file
        self.base_data_path = base_data_path
        imgs_labels, self.labels_dict, self.nr_classes = Aptos19_map_images_and_labels(base_data_path, label_file)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform


        return 
    
    
    # Method: __len__
    def __len__(self):
        return len(self.images_paths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        
        # Open image with PIL
        image = Image.open(os.path.join(self.base_data_path, img_name))
        #plt.imshow(image)
        
        # Perform transformations with Numpy Array
        image = np.asarray(image)
        
        #image = np.reshape(image, newshape=(image.shape[0], image.shape[1], 1))
        #image = np.concatenate((image, image, image), axis=2)

        # Load image with PIL
        image = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label


#=====================================================================================================
#======================================= CBIS 2020 ===================================================
#=====================================================================================================

# Function: Get images and labels from directory files
def map_images_and_labels(dir):
    # Images
    dir_files = os.listdir(dir)
    dir_imgs = [i for i in dir_files if i.split('.')[1]=='png']
    dir_imgs.sort()

    # Labels
    dir_labels_txt = [i.split('.')[0]+'case.txt' for i in dir_imgs]
    

    # Create a Numpy array to append file names and labels
    imgs_labels = np.zeros(shape=(len(dir_imgs), 2), dtype=object)

    # Go through images and labels
    idx = 0
    for image, label in zip(dir_imgs, dir_labels_txt):
        # Debug print
        # print(f"Image file: {image} | Label file: {label}")

        # Append image (Column 0)
        imgs_labels[idx, 0] = image
        
        # Append label (Column 1)
        # Read temp _label
        _label = np.genfromtxt(
            fname=os.path.join(dir, label),
            dtype=str
        )

        # Debug print
        # print(f"_label: {_label}")
        
        # Append to the Numpy Array
        imgs_labels[idx, 1] = str(_label)

        # Debug print
        # print(f"Image file: {imgs_labels[idx, 0]} | Label: {imgs_labels[idx, 1]}")


        # Update index
        idx += 1
    

    # Create labels dictionary to map strings into numbers
    _labels_unique = np.unique(imgs_labels[:, 1])

    # Nr of Classes
    nr_classes = len(_labels_unique)

    # Create labels dictionary
    labels_dict = dict()
    
    for idx, _label in enumerate(_labels_unique):
        labels_dict[_label] = idx


    return imgs_labels, labels_dict, nr_classes


# Create a Dataset Class
class CBISDataset(Dataset):
    def __init__(self, base_data_path, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.base_data_path = base_data_path
        imgs_labels, self.labels_dict, self.nr_classes = map_images_and_labels(dir=base_data_path)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        
        # Open image with PIL
        image = Image.open(os.path.join(self.base_data_path, img_name))
        
        # Perform transformations with Numpy Array
        image = np.asarray(image)
        image = np.reshape(image, newshape=(image.shape[0], image.shape[1], 1))
        image = np.concatenate((image, image, image), axis=2)

        # Load image with PIL
        image = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label
