# Imports
import numpy as np
import os
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# My Imports
from ModelLoops import train_model
from ModelLoops import active_train_model
from CustomDatasets import ISIC17_Dataset
from ModelArchitectures import PretrainedModel

# My Imports
from CustomDatasets import Aptos19_Dataset
from ModelArchitectures import PretrainedModel
from xAI_utils import GenerateDeepLift
from xAI_utils import GenerateDeepLiftAtts
from choose_rects import GenerateRectangles
from choose_rects import ChooseRectangles
from choose_rects import GetOracleFeedback


# CUDA
GPU_TO_USE="0"
device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# Data Directories
your_datasets_dir = "/home/pedro/Desktop"
data_dir = os.path.join(your_datasets_dir, "ISIC17")
data_name = "ISIC17"

#Model Directory
trained_models_dir = "/home/pedro/Desktop/retrained_models"

# train data
train_dir = os.path.join(data_dir, "train")
train_label_file = "ISIC-2017_Training_Part3_GroundTruth.csv"

# validation data
val_dir = os.path.join(data_dir, "val")
val_label_file = "ISIC-2017_Validation_Part3_GroundTruth.csv"

# train transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomAffine(degrees=(-180,180),translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.ColorJitter(brightness=0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])

# validation transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN,STD)
])

# Load and count data samples
train_fraction = 0.1
val_fraction = 1

# Load and count data samples
# Train Dataset
train_set = ISIC17_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms, transform_orig=val_transforms, fraction=train_fraction)
print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")

# Validation
val_set = ISIC17_Dataset(base_data_path=val_dir, label_file=val_label_file, transform=val_transforms, transform_orig=val_transforms,fraction=val_fraction)
print(f"Number of Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")

# get batch and build loaders
BATCH_SIZE = 10

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

nr_classes = 2
model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=2)
model_name = "efficientNet_b1"

# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#-------------------------------------
#------ LOAD STATION --------------
#-------------------------------------

# load previously trained model

trained_model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=2)
nr_classes = 2
model_path = "/home/pedro/Desktop/trained_models/efficientNet_b1_ISIC17/weights/efficientNet_b1_ISIC17_100p_100e.pt"
trained_model.load_state_dict(torch.load(model_path, map_location=DEVICE))


#=====================================================================================================
#======================================  xAI Experiments =============================================
#=====================================================================================================

# Define the Aptos Classes
classes = ('0', '1')

# Print and show images
#x_image_dir = "/home/up201605633/Desktop/Results/DeepLift/"

# get images from batch
dataiter = iter(val_loader)
images, images_og, labels, indices = dataiter.next()

for i in range(10):

    deepLiftAtts, query_pred = GenerateDeepLiftAtts(image=images_og[i], label=labels[i], model = trained_model, data_classes=classes)

    # Aggregate along color channels and normalize to [-1, 1]
    deepLiftAtts = deepLiftAtts.sum(axis=np.argmax(np.asarray(deepLiftAtts.shape) == 3))
    deepLiftAtts /= np.max(np.abs(deepLiftAtts))
    deepLiftAtts = torch.tensor(deepLiftAtts)
    print(deepLiftAtts.shape)

    x, y = GetOracleFeedback(image=images_og[i], label=labels[i], idx=indices[i], model_attributions=deepLiftAtts, pred=query_pred, rectSize=28, rectStride=28, nr_rects=10)




