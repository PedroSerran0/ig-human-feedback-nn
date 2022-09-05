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
from choose_rects import GetOracleFeedback

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
trained_models_dir = "/home/pedro/Desktop/trained_models"

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
val_fraction = 0.1

# Train Dataset
train_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=train_transforms, transform_orig=val_transforms, split='train', fraction = train_fraction)
print(f"Number of Total Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")
val_set = Aptos19_Dataset(base_data_path=train_dir, label_file=train_label_file, transform=val_transforms, transform_orig=val_transforms, split='test', fraction = val_fraction)
print(f"Number of Total Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")




# get batch and build loaders
BATCH_SIZE = 10
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

#-------------------------------------
#------ LOAD STATION --------------
#-------------------------------------

# load previously trained model

trained_model = PretrainedModel(pretrained_model="efficientnet_b1", n_outputs=5)
nr_classes = 5
model_path = "/home/pedro/Desktop/trained_models/efficientNet_b1_Aptos2019/weights/efficientNet_b1_50.0p_15e.pt"
trained_model.load_state_dict(torch.load(model_path, map_location=DEVICE))

#=====================================================================================================
#======================================  xAI Experiments =============================================
#=====================================================================================================

# Define the Aptos Classes
classes = ('0', '1', '2', '3', '4')

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

# #Generate Explanations for 10 batches (100 images)
# # for i in range(1):
# #     dataiter = iter(val_loader)
# #     GenerateBatchDeepLift(data_batch_iteration=dataiter, batch_it_size=10,model = model, data_classes=classes, save_file_dir=x_image_dir)


# def fig2img(fig):
#     """Convert a Matplotlib figure to a PIL Image and return it"""
#     import io
#     buf = io.BytesIO()
#     fig.savefig(buf)
#     buf.seek(0)
#     img = Image.open(buf)
#     return img

# # test idx
# test_idx = 8

# #fig = GenerateDeepLift(image = images[0], label=labels[0], data_classes=classes, model = model)
# att = GenerateDeepLiftAtts(image = images[test_idx], label=labels[test_idx], data_classes=classes, model = model)

# # Aggregate along color channels and normalize to [-1, 1]
# att = att.sum(axis=np.argmax(np.asarray(att.shape) == 3))
# att /= np.max(np.abs(att))
# att = torch.tensor(att)
# #print(att.shape)
# #plt.imshow(att, cmap = "seismic",clim=(-1,1))


# rectGenerator = GenerateRectangles(att, size=28, stride=28, nr_rects=5)
# rects = rectGenerator.get_ranked_patches()


# def imshow(img ,transpose = True):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()

# ogImage = images[test_idx]
# #trans = transforms.ToPILImage()
# #grayImage = trans(ogImage).convert('L')
# ogImage = torch.permute(ogImage,(1,2,0))
# #imshow(ogImage)

# ui = ChooseRectangles(ogImage,rects)
# ui.draw()
# plt.show()
# print(ui.selected)
# selected_rects = ui.get_selected_rectangles()
# print(selected_rects)


