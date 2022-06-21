from sklearn.metrics import accuracy_score
import torch
import os
import torchvision 
import torch.nn.functional as F
import numpy as np
import sklearn
from sklearn import model_selection
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# Captum Imports
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import LRP
from captum.attr import visualization as viz
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution

# My Imports
from CustomDatasets import Aptos19_Dataset
from ModelArchitectures import PretrainedModel
from choose_rects import ChooseRectangles

# Plt show function
def imshow(img ,transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# Function to get the explanations
def attribute_image_features(algorithm, input, labels, ind, **kwargs):
    tensor_attributions = algorithm.attribute(input, target=labels[ind], **kwargs)
    
    return tensor_attributions


def GenerateDeepLiftAtts(image, model, data_classes, label, temp_pred):
    #model.eval()

    # Get image classification
    model.to('cpu')
    # Add empty batch dimension (to allow the function to take in a single sample)
    image_batch = image[None]
    image_batch = image_batch.type('torch.FloatTensor') 
    outputs = model(image_batch)

    outputs_probs = torch.softmax(outputs,1)
    _, predicted = torch.max(outputs_probs, 1)
    pred = int(predicted[0])
    print('previous pred:', temp_pred, 'new pred:', pred, predicted.shape)

    trans = transforms.ToPILImage()
    trans2 = transforms.ToTensor()
    blur = torchvision.transforms.GaussianBlur((17,37), sigma=(0.9, 10.0))
    
    blurred_image = blur(image)#.filter(ImageFilter.BLUR)
#    r,g,b = blurred_image.split()
#    blurred_image = Image.merge('RGB', (r,g,b))
#    blurred_image=trans2(blurred_image)
    #imshow(blurred_image)

    input = image.unsqueeze(0)
    #input.requires_grad = True
    
    # Generate the visual explnations (saliency maps)
    # DeepLift
    
    deeplift = DeepLift(model)
    
    # Reset model's gradients
    model.zero_grad()
    input = input.type('torch.FloatTensor') 
    
     #set ones as reference
    #reference = torch.ones(3,224,224)
    #reference = (reference.unsqueeze(0)).type('torch.FloatTensor')
    
    reference = (blurred_image.unsqueeze(0)).type('torch.FloatTensor')

    #dl_att = deeplift.attribute(input, target=label.item())
    dl_att = deeplift.attribute(input, target=pred, baselines=reference)
#    relu = torch.nn.ReLU()
#    dl_att = relu(dl_att)
    dl_att = np.transpose(dl_att.squeeze().cpu().detach().numpy(), (1, 2, 0))
 
    #imshow(dl_att)
    
    original_image = np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

     # Visualization of deep lift attributions
#    deepLiftFig,_ = viz.visualize_image_attr(dl_att, original_image=original_image,
#                                     method= "heat_map",
#                                     sign="all",
#                                     title="Deep Lift Attribution")


    return dl_att, pred

def GenerateDeepLift(image, model, data_classes, label):
    model.eval()

    # Get image classification
    model.to('cpu')
    # Add empty batch dimension (to allow the function to take in a single sample)
    image_batch = image[None]
    image_batch = image_batch.type('torch.FloatTensor') 
    outputs = model(image_batch)

    _, predicted = torch.max(outputs, 1)

    input = image.unsqueeze(0)
    input.requires_grad = True
    
    # Generate the visual explnations (saliency maps)
    # DeepLift
    deeplift = DeepLift(model)

    # Reset model's gradients
    model.zero_grad()
    input = input.type('torch.FloatTensor') 
    dl_att = deeplift.attribute(input, target=label.item())
    dl_att = np.transpose(dl_att.squeeze().cpu().detach().numpy(), (1, 2, 0))
    


    original_image = np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    # overlayed integrated gradients figure
    deepLiftFig,_ = viz.visualize_image_attr_multiple(dl_att, original_image, ["original_image","blended_heat_map"],
                                        ["all","all"],
                                        show_colorbar=True,
                                        titles=[f"Original(gt = {data_classes[label]}, predicted = {data_classes[predicted]})", "DeepLift"],
                                        fig_size=(10, 6))


    return deepLiftFig


def GenerateGradCamBatch(data_batch_iteration,batch_it_size, model, data_classes, save_file_dir):
    images, labels = data_batch_iteration.next()
    print('GroundTruth: ', ' '.join('%5s' % data_classes[labels[j]] for j in range(batch_it_size)))

    # Get image classification
    model.to('cpu')
    outputs = model(images)

    _, predicted = torch.topk(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % data_classes[predicted[j]] for j in range(batch_it_size)))

    # Setup
    layer_gradcam = LayerGradCam(model, model.model[0,1])
    ind = 0

    for image in images:
        input = image.unsqueeze(0)
        input.requires_grad = True

        attributions_lgc = layer_gradcam.attribute(input, target=predicted[ind])

        # Visualization of conv layer
        _ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                    sign="all",
                                    title="Last conv layer")
        # Upscalling
        upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input.shape[2:])

        print(attributions_lgc.shape)
        print(upsamp_attr_lgc.shape)
        print(image.shape)

        # Visualization of attributions
        _ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                            image.squeeze().permute(1,2,0).numpy(),
                                            ["original_image","blended_heat_map","masked_image"],
                                            ["all","positive","positive"],
                                            show_colorbar=True,
                                            titles=["Original", "Positive Attribution", "Masked"],
                                            fig_size=(18, 6))

 
    return

# Sort aux function: take third element for sort
def takeThird(elem):
    return elem[2]
