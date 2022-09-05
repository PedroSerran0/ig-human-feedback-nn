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



# ROSE Dataset
# Class: ROSE Dataset
class ROSE_Dataset(Dataset):
    def __init__(self, base_data_path, data_split, attack_type=None, transform=None, transform_orig=None):
        """
        Args:
            base_data_path (string): Data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
        # Init variables
        self.base_data_path = base_data_path
        self.data_split = data_split
        self.attack_type = attack_type
        imgs_labels, self.labels_dict, self.nr_classes = self.map_images_and_labels(base_data_path, data_split, attack_type=attack_type)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform
        self.transform_orig = transform_orig

        return
    

    # Method: Get images of a specific split and class
    def get_images(self, data_split, img_class, attack_type=None, data_path='/home/up201605633/Desktop/ROSE/data_divided'):

        # Assert conditions
        # Data split
        assert data_split in ('train', 'test'), f"Data split should be 'train' or 'test. You entered {data_split}."

        # Image class
        assert img_class in (0, 1), f"Image class should be either 0 (bonafide) or 1 (attack). You entered {img_class}."



        # Enter data_split directory
        data_split_dir = os.path.join(data_path, data_split)

        # Enter img_class directory
        img_class_dir = os.path.join(data_split_dir, str(img_class))


        # If img_class is 0, we return all the images there
        if img_class == 0:
            images = [i for i in os.listdir(img_class_dir) if not i.startswith('.')]

        else:
            # Assess the type of attack
            if attack_type is not None:
                assert attack_type in [i for i in range(1, 8)], f"Attack type should be an integer between 1-7. You entered {attack_type}."
                
                # Get the images
                images = [i for i in os.listdir(img_class_dir) if not i.startswith('.')]
                images = [i for i in images if int(i.split('_')[2])==int(attack_type)]


            else:
                images = [i for i in os.listdir(img_class_dir) if not i.startswith('.')]


        return images


    # Method: Map images and labels
    def map_images_and_labels(self, data_dir, data_split, attack_type):
        # Get attack images
        attackImages = self.get_images(data_split=data_split, img_class=1, attack_type=attack_type, data_path=data_dir)
        attackLabels = np.ones(len(attackImages))
        attackData = np.column_stack((attackImages, attackLabels))

        # Get genuine images
        genuineImages = self.get_images(data_split=data_split, img_class=0, data_path=data_dir)
        genuineLabels = np.zeros(len(genuineImages))
        genuineData = np.column_stack((genuineImages, genuineLabels))

        allData = np.concatenate((attackData, genuineData))
        
        # Get nr classes
        _labels_unique = np.unique(allData[:,1])
        nr_classes = len(_labels_unique)

        # Create labels dictionary
        labels_dict = dict()
            
        for idx, _label in enumerate(_labels_unique):
            labels_dict[_label] = idx

        return allData, labels_dict, nr_classes


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
        image = Image.open(os.path.join(self.base_data_path, self.data_split, str(self.labels_dict[self.images_labels[idx]]),img_name))
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



# CBIS Dataset
# Class: CBIS Dataset
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
        imgs_labels, self.labels_dict, self.nr_classes = self.map_images_and_labels(directory=base_data_path)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform


        return 


    # Method: Map images and labels
    def map_images_and_labels(self, directory):
        
        # Images
        dir_files = os.listdir(directory)
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
                fname=os.path.join(directory, label),
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
