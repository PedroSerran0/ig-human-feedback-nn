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
from CustomDatasets import ISIC17_Dataset
from ModelArchitectures import PretrainedModel


# CUDA
GPU_TO_USE="0"
device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# Data Directories
your_datasets_dir = "/home/up201605633/Desktop"
data_dir = os.path.join(your_datasets_dir, "ISIC17")
data_name = "ISIC17"

#Model Directory
trained_models_dir = "/home/up201605633/Desktop/trained_models"

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
# Train Dataset
train_set = ISIC17_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms)
print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")

# Validation
val_set = ISIC17_Dataset(base_data_path=val_dir, label_file=val_label_file, transform=val_transforms)
print(f"Number of Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")

# get batch and build loaders
BATCH_SIZE = 10

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Set resNet50 path
resNet50_dir = os.path.join(trained_models_dir, "denseNet121_ISIC17")

# Results and Weights
weights_dir = os.path.join(resNet50_dir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(resNet50_dir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)
    
# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

nr_classes = 2
model = PretrainedModel(pretrained_model="densenet121", n_outputs=2)
model_name = "denseNet121"

# Hyper-parameters
EPOCHS = 200
class_weight = torch.tensor([1/(2*0.8), 1/(2*0.2)])  # passar para parametro
class_weight = class_weight.to(DEVICE)
LOSS = torch.nn.CrossEntropyLoss(weight=class_weight)
#LOSS = torch.nn.CrossEntropyLoss()

val_losses,train_losses,val_metrics,train_metrics = train_model(model=model, model_name=model_name,nr_classes=nr_classes,train_loader=train_loader,
                 val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name = data_name,
                    LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE)

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss (ISIC17-DenseNet121)")
plt.plot(val_losses,label="val-loss")
plt.plot(train_losses,label="train-loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,"ISIC17_DenseNet121_loss_100E_2.png"))
plt.show()


plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy (ISIC17-DenseNet121)")
plt.plot(val_metrics[:,0], label = "val-acc")
plt.plot(train_metrics[:,0], label="train-acc")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,"ISIC17_DenseNet121_acc_100E_2.png"))
plt.show()