# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:45:51 2022

@author: Fnac Home
"""

# Imports
from selectors import EpollSelector
import numpy as np
import os
from PIL import Image
import sklearn
from sklearn import model_selection

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
from CustomDatasets import Aptos19_Dataset
from ModelArchitectures import PretrainedModel

# CUDA
GPU_TO_USE="0"
device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Data Directories
your_datasets_dir = "/home/pedro/Desktop"
data_name = "Aptos2019"
data_dir = os.path.join(your_datasets_dir, data_name)


#Model Directory
trained_models_dir = "/home/pedro/Desktop/retrained_models"

# train data
train_dir = os.path.join(data_dir, "train")
train_label_file = "train.csv"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomAffine(degrees=(-180,180),translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.ColorJitter(brightness=0.3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])

#validation transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN,STD)
])

# Load and count data samples
train_fraction = 0.1
val_fraction = 1

# Train Dataset
train_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms, transform_orig=val_transforms, split='train', fraction = train_fraction)
print(f"Number of Total Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")
val_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=val_transforms, transform_orig=val_transforms, split='test', fraction = val_fraction)
print(f"Number of Total Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")




# get batch and build loaders
BATCH_SIZE = 4
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=5)
model_name = "efficientNet_b1"

# Set model path
trained_model_name = f"{model_name}_{data_name}"
model_dir = os.path.join(trained_models_dir, trained_model_name)

# Results and Weights
weights_dir = os.path.join(model_dir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(model_dir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)
    
# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Hyper-parameters
EPOCHS = 20
LOSS = torch.nn.CrossEntropyLoss()

# Active Learning parameters
entropy_thresh = 0
nr_queries = 20
data_classes = ('0', '1', '2', '3', '4')
start_epoch = 1
percentage = train_fraction*100

#val_losses,train_losses,val_metrics,train_metrics = active_train_model(model=model, model_name=model_name, data_name=data_name, train_loader=train_loader, val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir,
#                                                entropy_thresh=entropy_thresh, nr_queries=nr_queries, data_classes=data_classes, start_epoch = start_epoch, percentage = percentage,
#                                                 EPOCHS=EPOCHS, DEVICE=DEVICE, LOSS=LOSS)


#val_losses,train_losses,val_metrics,train_metrics = train_model(model=model, model_name=model_name,nr_classes=5,train_loader=train_loader,
#                  val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name=data_name,
#                     LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE, percentage=percentage)




#Load and Retrain
trained_model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=5)
trained_model_name = "efficientNet_b1_10E_retrained"
nr_classes = 5
model_path = "/home/pedro/Desktop/trained_models/efficientNet_b1_Aptos2019/weights/efficientNet_b1_Aptos2019_10.0p_10e.pt"
trained_model.load_state_dict(torch.load(model_path, map_location=DEVICE))



val_losses,train_losses,val_metrics,train_metrics = active_train_model(model=trained_model, model_name=trained_model_name, data_name=data_name, train_loader=train_loader, val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir,
                                               entropy_thresh=entropy_thresh, nr_queries=nr_queries, data_classes=data_classes, start_epoch = start_epoch, percentage = percentage,
                                                EPOCHS=EPOCHS, DEVICE=DEVICE, LOSS=LOSS)

#val_losses,train_losses,val_metrics,train_metrics = train_model(model=trained_model, model_name=trained_model_name,nr_classes=5,train_loader=train_loader,
#                  val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name=data_name,
#                     LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE, percentage=percentage)


plt.figure(figsize=(10,5))
plt.title(f"Training and Validation Metrics ({trained_model_name}_AL_{percentage}%)")
plt.plot(val_losses,label="val-loss", linestyle='--')
plt.plot(train_losses,label="train-loss")
plt.plot(val_metrics[:,0], label = "val-acc", linestyle='--')
plt.plot(train_metrics[:,0], label="train-acc")
plt.xlabel("Iterations")
plt.ylabel("Metrics")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_metrics_{EPOCHS}E__{percentage}p.png"))
plt.show()

print("plot saved")

# plt.figure(figsize=(10,5))
# plt.title(f"Training and Validation Accuracy ({trained_model_name}_AL)")
# plt.plot(val_metrics[:,0], label = "val-acc")
# plt.plot(train_metrics[:,0], label="train-acc")
# plt.xlabel("Iterations")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_acc_{EPOCHS}E_AL.png"))
# plt.show()

# print("accuracy plot saved")


