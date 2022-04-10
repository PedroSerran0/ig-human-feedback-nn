# Imports
from matplotlib import mlab
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
from ModelLoops import test_model
from CustomDatasets import ISIC17_Dataset
from ModelArchitectures import ResNet50

# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Data Directories
your_datasets_dir = "/home/up201605633/Desktop"
data_dir = os.path.join(your_datasets_dir, "ISIC17")

# test data
test_dir = os.path.join(data_dir, "test")
test_label_file = "ISIC-2017_Test_v2_Part3_GroundTruth.csv"

# Model Directory
trained_models_dir = "/home/up201605633/Desktop/trained_models"
# Set resNet50 path
resNet50_dir = os.path.join(trained_models_dir, "resNet50_ISIC17")
trained_models_dir = "/home/up201605633/Desktop/trained_models"

# Results and Weights
weights_dir = os.path.join(resNet50_dir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)

# Test transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN,STD)
])

# Test
test_set = ISIC17_Dataset(base_data_path=test_dir, label_file=test_label_file, transform=test_transforms)
print(f"Number of Validation Images: {len(test_set)} | Label Dict: {test_set.labels_dict}")

# get batch and build loaders
BATCH_SIZE = 10
LOSS = torch.nn.CrossEntropyLoss()
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

nr_classes = 2
model = ResNet50(pretrained_model="resnet50", n_outputs=2)
model_name = "resNet50"

model_path = os.path.join(weights_dir, f"{model_name}_ISIC.pt")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))

test_loss, test_metrics = test_model(model, model_name, test_loader, nr_classes, LOSS, DEVICE)