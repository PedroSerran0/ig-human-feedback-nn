# Imports
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import pandas as pd
from skimage.io import imread

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Custom definitions
ImageFile.LOAD_TRUNCATED_IMAGES = True



# Function: Resize images based on new height
def resize_images(datapath, newpath, new_height=512):

    # Create new directories if needed
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    

    # Go through the data
    for f in tqdm(os.listdir(datapath)):
        if(f.endswith(".jpg") or f.endswith('.png')):
            img = Image.open(os.path.join(datapath, f))
            w, h = img.size
            ratio = w / h
            new_w = int(np.ceil(new_height * ratio))
            new_img = img.resize((new_w, new_height), Image.ANTIALIAS)
            new_img.save(os.path.join(newpath, f))

    return



# NCI Dataset
# Class: NCI Dataset 
class NCI_Dataset(Dataset):
    def __init__(self, fold, path, transform=None, transform_orig=None, fraction=1):
        assert fold in ['train', 'test']
        # self.root = '/data/NCI/training/NHS'
        self.root = path
        self.files = [f for f in os.listdir(self.root) if f.endswith('_C1.jpg')]
        rand = np.random.RandomState(123)
        ix = rand.choice(len(self.files), len(self.files), False)
        ix = ix[:int(0.75*len(ix))] if fold == 'train' else ix[int(0.75*len(ix)):]
        self.files = [self.files[i] for i in ix]
        df = pd.read_excel(os.path.join(self.root, 'covariate_data_training_NHS.xls'), skiprows=2)
        self.classes = [df['WRST_HIST_AFTER'][df['IMAGE_ID'] == f].iloc[0] for f in self.files]
        self.transform = transform
        self.transform_orig = transform_orig
        
        # get desired fraction of data
        data_size = len(self.files)
        target_size = round(fraction * data_size)
        self.files = self.files[0:(target_size-1)]


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        image = imread(os.path.join(self.root, self.files[idx]))
        label = self.classes[idx]
        label = 0 if label <= 0 else 1
        
        image = np.asarray(image)
        
        # Load image with PIL
        image = image_orig = Image.fromarray(image)

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        if self.transform_orig:
            image_orig = self.transform_orig(image_orig)


        return image, image_orig, label, idx



# ISIC2017 Dataset
# Class: ISIC2017 Dataset
class ISIC17_Dataset(Dataset):
    def __init__(self, base_data_path, label_file, transform=None, transform_orig=None, fraction=1):
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
        imgs_labels, self.labels_dict, self.nr_classes = self.map_images_and_labels(base_data_path, label_file)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform
        self.transform_orig = transform_orig
        
        # get desired fraction of data
        data_size = len(imgs_labels)
        target_size = round(fraction * data_size)
        imgs_labels = imgs_labels[0:(target_size-1)]

        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]


        return 


    # Method: Map images and labels
    def map_images_and_labels(self, data_dir, label_file):
        
        # Get image_id and corresponding label from csv file
        labels = np.genfromtxt(os.path.join(data_dir, label_file), delimiter=',',encoding="utf8", dtype=None)
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
        image = image_orig = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        if self.transform_orig:
            image_orig = self.transform_orig(image_orig)

        #print(img_name, idx)
        return image, image_orig, label, idx



# APTOS2019 Dataset
# Class: APTOS2019 Dataset
class Aptos19_Dataset(Dataset):
    def __init__(self, base_data_path, label_file, transform=None, transform_orig=None, split= 'train', fraction=1):
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
        imgs_labels, self.labels_dict, self.nr_classes = self.map_images_and_labels(base_data_path, label_file)

        # split train/test
        assert split in ['train', 'test']
        rand = np.random.RandomState(123)
        ix = rand.choice(len(imgs_labels), len(imgs_labels), False)
        if split == 'train':
            ix = ix[:int(len(ix)*0.8)]
        else:
            ix = ix[int(len(ix)*0.8):]
        imgs_labels = imgs_labels[ix]

        # get desired fraction of data
        data_size = len(imgs_labels)
        target_size = round(fraction * data_size)
        imgs_labels = imgs_labels[0:(target_size-1)]

        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform
        self.transform_orig = transform_orig

        return


    # Method: Map images and labels
    def map_images_and_labels(self, data_dir, label_file):
        
        # Get image_id and corresponding label from csv file
        labels = np.genfromtxt(os.path.join(data_dir, label_file), delimiter=',',encoding="utf8", dtype=None)
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
        image = image_orig = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        if self.transform_orig:
            image_orig = self.transform_orig(image_orig)

        #print(img_name, idx)
        return image, image_orig, label, idx