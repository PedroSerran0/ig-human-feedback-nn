from ctypes import sizeof
from sklearn.metrics import accuracy_score
import torch
import os
import torchvision 
import torch.nn.functional as F
import numpy as np
import sklearn
from sklearn import model_selection
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import skimage

# Captum Imports
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import LRP
from captum.attr import visualization as viz

# My Imports
from CustomDatasets import Aptos19_Dataset
from ModelArchitectures import PretrainedModel
from xAI_utils import GenerateDeepLift
from xAI_utils import GenerateDeepLiftAtts
from choose_rects import GenerateRectangles
from choose_rects import ChooseRectangles

# CUDA
GPU_TO_USE="0"
device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Data Directories
your_datasets_dir = "/home/up201605633/Desktop"
data_name = "Aptos2019"
data_dir = os.path.join(your_datasets_dir, data_name)

#Model Directory
trained_models_dir = "/home/up201605633/Desktop/trained_models"

# train data
train_dir = os.path.join(data_dir, "train")
train_label_file = "train.csv"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
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
# Train Dataset
train_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms)
print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")
val_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=val_transforms)

# Set target train and val sizes
val_size = 0.2 # portion of the dataset
num_train = len(train_set)
indices = list(range(num_train))
split_idx = int(np.floor(0.2 * num_train))

train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
assert len(train_idx) != 0 and len(valid_idx) != 0

# Split the train set into train and val
train_indices, val_indices = sklearn.model_selection.train_test_split(indices)
train_set = torch.utils.data.Subset(train_set, train_indices)
val_set = torch.utils.data.Subset(val_set, val_indices)

# get batch and build loaders
BATCH_SIZE = 10
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

nr_classes = 5

# Define model
model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=5)
model_name = "efficientNet_b1"

# Set model path
trained_model_name = f"{model_name}_{data_name}"
model_dir = os.path.join(trained_models_dir, trained_model_name)
weights_dir = os.path.join(model_dir, "weights")
history_dir = os.path.join(model_dir, "history")

# Load model
model_path = os.path.join(weights_dir, f"{model_name}_{data_name}.pt")
model.load_state_dict(torch.load(model_path, map_location = device))
model.eval()

#=====================================================================================================
#======================================  xAI Experiments =============================================
#=====================================================================================================

# Define the Aptos Classes
classes = ('0', '1', '2', '3', '4')

# Print and show images
x_image_dir = "/home/up201605633/Desktop/Results/DeepLift/"

# get images from batch
dataiter = iter(val_loader)
images, labels = dataiter.next()

#Generate Explanations for 10 batches (100 images)
# for i in range(1):
#     dataiter = iter(val_loader)
#     GenerateBatchDeepLift(data_batch_iteration=dataiter, batch_it_size=10,model = model, data_classes=classes, save_file_dir=x_image_dir)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# test idx
test_idx = 8

#fig = GenerateDeepLift(image = images[0], label=labels[0], data_classes=classes, model = model)
att = GenerateDeepLiftAtts(image = images[test_idx], label=labels[test_idx], data_classes=classes, model = model)

# Aggregate along color channels and normalize to [-1, 1]
att = att.sum(axis=np.argmax(np.asarray(att.shape) == 3))
att /= np.max(np.abs(att))
att = torch.tensor(att)
#print(att.shape)
#plt.imshow(att, cmap = "seismic",clim=(-1,1))


# rectGenerator = GenerateRectangles(att, size=28, stride=28, nr_rects=5)
# rects = rectGenerator.get_ranked_patches()


def imshow(img ,transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

ogImage = images[test_idx]
#trans = transforms.ToPILImage()
#grayImage = trans(ogImage).convert('L')
#ogImage = torch.permute(ogImage,(1,2,0))
imshow(ogImage)

# ui = ChooseRectangles(ogImage,rects)
# ui.draw()
# plt.show()
# print(ui.selected)
