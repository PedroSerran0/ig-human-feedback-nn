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
BATCH_SIZE = 1

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

nr_classes = 2
model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=2)
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
# Active Learning parameters
entropy_thresh = 0
nr_queries = 20
data_classes = ('0', '1')
start_epoch = 1
percentage = train_fraction*100
#class_weight = torch.tensor([1/(2*0.8), 1/(2*0.2)])  # passar para parametro
#class_weight = class_weight.to(DEVICE)
#LOSS = torch.nn.CrossEntropyLoss(weight=class_weight)
LOSS = torch.nn.CrossEntropyLoss()

#val_losses,train_losses,val_metrics,train_metrics = active_train_model(model=model, model_name=model_name, data_name=data_name, train_loader=train_loader, val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir,
#                                                entropy_thresh=entropy_thresh, nr_queries=nr_queries, data_classes=data_classes, start_epoch = start_epoch, percentage = percentage,
#                                                 EPOCHS=EPOCHS, DEVICE=DEVICE, LOSS=LOSS)


##val_losses,train_losses,val_metrics,train_metrics = train_model(model=model, model_name=model_name,nr_classes=nr_classes,train_loader=train_loader,
##                  val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name=data_name,
##                     LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE, percentage=percentage)

#plt.figure(figsize=(10,5))
#plt.title(f"Training and Validation Metrics ({trained_model_name}_{EPOCHS}E_AL_{percentage}%)")
#plt.plot(val_losses,label="val-loss", linestyle='--', color="green")
#plt.plot(train_losses,label="train-loss", color="green")
#plt.plot(val_metrics[:,0], label = "val-acc", linestyle='--', color="red")
#plt.plot(train_metrics[:,0], label="train-acc", color="red")
#plt.xlabel("Iterations")
#plt.ylabel("Metrics")
#plt.legend()
#plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_metrics_{EPOCHS}E_AL_{percentage}p.png"))
#plt.show()

#print("plot saved")

#-------------------------------------
#------ RETRAIN STATION --------------
#-------------------------------------

# load previously trained model
train_description = "5AUTO_re_20AUTO"
pretrained_epochs = 5
trained_model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=2)
trained_model_name = f"{data_name}_efficientNet_b1_{train_description}"
nr_classes = 2
model_path = "/home/pedro/Desktop/trained_models/efficientNet_b1_ISIC17/weights/efficientNet_b1_ISIC17_10.0p_5e.pt"
trained_model.load_state_dict(torch.load(model_path, map_location=DEVICE))


#val_losses,train_losses,val_metrics,train_metrics = active_train_model(model=trained_model, model_name=trained_model_name, data_name=data_name, train_loader=train_loader, val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir,
#                                               entropy_thresh=entropy_thresh, nr_queries=nr_queries, data_classes=data_classes, oversample=True, start_epoch=start_epoch, percentage=percentage,
#                                                EPOCHS=EPOCHS, DEVICE=DEVICE, LOSS=LOSS)

val_losses,train_losses,val_metrics,train_metrics = train_model(model=trained_model, model_name=trained_model_name,nr_classes=nr_classes,train_loader=train_loader,
                  val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name=data_name,
                     LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE, percentage=percentage)


plt.figure(figsize=(10,5))
plt.title(f"{train_description} and Validation Metrics ({trained_model_name}_{percentage}%)")
plt.plot(val_losses,label="val-loss", linestyle='--', color="green")
plt.plot(train_losses,label="train-loss", color="green")
plt.plot(val_metrics[:,0], label = "val-acc", linestyle='--',color="red")
plt.plot(train_metrics[:,0], label="train-acc",color="red")
plt.xlabel("Iterations")
plt.ylabel("Metrics")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_metrics_{percentage}p.png"))
plt.show()

print("plot saved")






