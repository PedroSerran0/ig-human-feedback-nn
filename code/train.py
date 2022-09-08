# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# PyTorch Imports
import torch
import torchvision

# Project Imports
from data_utilities import Aptos19_Dataset, ISIC17_Dataset, NCI_Dataset
from model_architectures import PretrainedModel
from model_loops import train_model, active_train_model

parser = argparse.ArgumentParser(description='Run HITL training')

# Directories
parser.add_argument('-dr', '--data_dir', type=str, metavar='', required=True, help='Train Data Directory')
parser.add_argument('-md', '--models_dir', type=str, metavar='', required=True, help='Trained Models Directory')

# Training Hyperparameters
parser.add_argument('-E', '--epochs', type=int, metavar='', required=True, help='Training Epochs')
parser.add_argument('-tf', '--tr_fraction', type=float, metavar='', required=True, help='Fraction of train data')
parser.add_argument('-vf', '--val_fraction', type=float, metavar='', required=True, help='Fraction of val data')
parser.add_argument('-td', '--train_desc', type=str, metavar='', required=True, help='Train title')

# HITL Options
parser.add_argument('-sp', '--sampling', type=str, choices=['low_entropy', 'high_entropy'], metavar='', required=True, help='Sampling Process (low_entropy or high_entropy)')
parser.add_argument('-et', '--entropy_thresh', type=float, metavar='', required=True, help='Entropy Threshold')
parser.add_argument('-qu', '--nr_queries', type=int, metavar='', required=True, help='Number of Queries per epoch')
parser.add_argument('-ov', '--isOversampled', type=bool, metavar='', required=True, help='Activate/Deactivate oversampling')
parser.add_argument('-se', '--start_epoch', type=int, metavar='', required=True, help='HITL activation epoch')

# Dataset
parser.add_argument('-ds', '--dataset', type=str, choices=['APTOS19', 'ISIC17','NCI'], metavar='', required=True, help='Define the dataset (APTOS, ISIC,NCI)')

args = parser.parse_args()

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# CUDA
GPU_TO_USE="0"
device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# Data Directories
#your_datasets_dir = "data"
your_datasets_dir = args.data_dir
data_name = args.dataset
#data_dir = os.path.join(your_datasets_dir, data_name)
data_dir = your_datasets_dir


# Model Directory
#trained_models_dir = "results/ones_test"
trained_models_dir = args.models_dir

# Load and count data samples
train_fraction = args.tr_fraction
val_fraction = args.val_fraction

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomAffine(degrees=(-180,180),translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.ColorJitter(brightness=0.3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])


# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN,STD)
])


if args.dataset == "APTOS19":
    # Train data
    train_dir = os.path.join(data_dir, "train")
    train_label_file = "train.csv"
    data_classes = ('0', '1', '2', '3', '4')
    # Train Dataset
    train_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms, transform_orig=val_transforms, split='train', fraction = train_fraction)
    print(f"Number of Total Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")
    val_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=val_transforms, transform_orig=val_transforms, split='test', fraction = val_fraction)
    print(f"Number of Total Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")
elif args.dataset == "ISIC17":
    # Train data
    train_dir = os.path.join(data_dir, "train")
    train_label_file = "ISIC-2017_Training_Part3_GroundTruth.csv"
    data_classes = ('0', '1')
    # Validation data
    val_dir = os.path.join(data_dir, "val")
    val_label_file = "ISIC-2017_Validation_Part3_GroundTruth.csv"
    # Load and count data samples
    # Train Dataset
    train_set = ISIC17_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms, transform_orig=val_transforms, fraction=train_fraction)
    print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")

    # Validation
    val_set = ISIC17_Dataset(base_data_path=val_dir, label_file=val_label_file, transform=val_transforms, transform_orig=val_transforms,fraction=val_fraction)
    print(f"Number of Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")
elif args.dataset == "NCI":
    # Train Dataset
    data_classes = ('0', '1')
    train_set = NCI_Dataset(fold="train", path=data_dir, transform=train_transforms, transform_orig=val_transforms, fraction=train_fraction)
    print(f"Number of Total Train Images: {len(train_set)}")
    val_set = NCI_Dataset(fold="test", path=data_dir, transform=val_transforms, transform_orig=val_transforms, fraction=val_fraction)
    print(f"Number of Total Validation Images: {len(val_set)}")


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
EPOCHS = args.epochs
LOSS = torch.nn.CrossEntropyLoss()

# Active Learning parameters
entropy_thresh = args.entropy_thresh
nr_queries = args.nr_queries
start_epoch = args.start_epoch
percentage = train_fraction*100
isOversampled = args.isOversampled
sampling_process = args.sampling

train_description = "auto_100_lr5_ones"

val_losses, train_losses, val_metrics, train_metrics = active_train_model(
    model=model,
    model_name=model_name,
    train_loader=train_loader,
    val_loader=val_loader,
    history_dir=history_dir,
    weights_dir=weights_dir,
    entropy_thresh=entropy_thresh,
    nr_queries=nr_queries,
    data_classes=data_classes,
    oversample=isOversampled,
    sampling_process=sampling_process,
    start_epoch=start_epoch,
    percentage=percentage,
    EPOCHS=EPOCHS,
    DEVICE=DEVICE,
    LOSS=LOSS
)


#val_losses,train_losses,val_metrics,train_metrics = train_model(model=model, model_name=model_name,nr_classes=5,train_loader=train_loader,
#                  val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name=data_name,
#                     LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE, percentage=percentage)


#-------------------------------------
#------ RETRAIN STATION --------------
#-------------------------------------

# load previously trained model
#train_description = "200base_reAL_20E_lr4"
#pretrained_epochs = 200
#trained_model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=5)
#trained_model_name = f"{data_name}_efficientNet_b1_{train_description}"
#nr_classes = 5
#model_path = "/home/pedro/Desktop/trained_models/efficientNet_b1_Aptos2019/weights/efficientNet_b1_100p_200e.pt"
#trained_model.load_state_dict(torch.load(model_path, map_location=DEVICE))


#val_losses,train_losses,val_metrics,train_metrics = active_train_model(model=trained_model, model_name=trained_model_name, data_name=data_name, train_loader=train_loader, val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir,
#                                               entropy_thresh=entropy_thresh, nr_queries=nr_queries, data_classes=data_classes, oversample=True, start_epoch = start_epoch, percentage = percentage,
#                                                EPOCHS=EPOCHS, DEVICE=DEVICE, LOSS=LOSS)

#val_losses,train_losses,val_metrics,train_metrics = train_model(model=trained_model, model_name=trained_model_name,nr_classes=nr_classes,train_loader=train_loader,
#                  val_loader=val_loader, history_dir=history_dir, weights_dir=weights_dir, data_name=data_name,
#                     LOSS=LOSS, EPOCHS=EPOCHS, DEVICE=DEVICE, percentage=percentage)


plt.figure(figsize=(10,5))
plt.title(f"{train_description} Accuracy and Loss ({trained_model_name}_{percentage}%)")
plt.plot(val_losses,label="val-loss", linestyle='--', color="green")
plt.plot(train_losses,label="train-loss", color="green")
plt.plot(val_metrics[:,0], label = "val-acc", linestyle='--',color="red")
plt.plot(train_metrics[:,0], label="train-acc",color="red")
plt.xlabel("Iterations")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_{train_description}_metrics_{percentage}p.png"))
plt.show()

print("plot saved")

plt.figure(figsize=(10,5))
plt.title(f"{train_description} Recall, Precision,F1 ({trained_model_name}_{percentage}%)")
plt.plot(val_metrics[:,1],label="val-recall", linestyle='--', color="green")
plt.plot(train_metrics[:,1],label="train-recall", color="green")
plt.plot(val_metrics[:,2], label = "val-precision", linestyle='--',color="red")
plt.plot(train_metrics[:,2], label="train-precision",color="red")
plt.plot(val_metrics[:,3], label = "val-f1", linestyle='--',color="blue")
plt.plot(train_metrics[:,3], label="train-f1",color="blue")
plt.xlabel("Iterations")
plt.ylabel("Metrics")
plt.legend()
plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_{train_description}_metrics2_{percentage}p.png"))
plt.show()

print("plot saved")
