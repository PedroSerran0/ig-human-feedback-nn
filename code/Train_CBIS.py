# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
import torchvision

# Project Imports
from data_utilities import CBISDataset
from model_architectures import resnet50
from model_loops import train_model



# CUDA
GPU_TO_USE="0"
device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)



# Data Directories
your_datasets_dir = "data"
data_dir = os.path.join(your_datasets_dir, "CBISPreprocDataset")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")


# Model Directory
trained_models_dir = "results/trained_models"


# Data normalisations parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Train transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])


# Validation transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])


# Load and count data samples
# Train Dataset
train_set = CBISDataset(base_data_path=train_dir, transform=train_transforms)
print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")

# Validation
val_set = CBISDataset(base_data_path=val_dir,transform=val_transforms)
print(f"Number of Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")

# Test
test_set = CBISDataset(base_data_path=test_dir,transform=val_transforms)
print(f"Number of Test Images: {len(test_set)} | Label Dict: {test_set.labels_dict}")

# get batch and build loaders
BATCH_SIZE = 10

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Set resNet50 path
resNet50_dir = os.path.join(trained_models_dir, "resNet50_CBIS")

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

# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Input Data Dimensions
img_nr_channels = 3
img_height = 224
img_width = 224

# ResNet50
nr_classes = 2
model = resnet50(pretrained_model="resnet50", n_outputs=2)
model_name = "resNet50"

# Hyper-parameters
EPOCHS = 50
LOSS = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 1e-4

val_losses, train_losses, val_metrics, train_metrics = train_model(
    model=model,
    model_name=model_name,
    train_loader=train_loader,
    val_loader=val_loader,
    history_dir=history_dir,
    weights_dir=weights_dir,
    LOSS=LOSS,
    EPOCHS=EPOCHS,
    DEVICE=DEVICE
)


plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss (CBIS-ResNet50)")
plt.plot(val_losses,label="val-loss")
plt.plot(train_losses,label="train-loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,"CBIS_resnet50_loss.png"))
plt.show()


plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy (CBIS-ResNet50)")
plt.plot(val_metrics[:,0], label = "val-acc")
plt.plot(train_metrics[:,0], label="train-acc")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,"CBIS_resnet50_acc.png"))
plt.show()
