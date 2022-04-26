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
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

# Function to get the explanations
def attribute_image_features(algorithm, input, labels, ind, **kwargs):
    tensor_attributions = algorithm.attribute(input, target=labels[ind], **kwargs)
    
    return tensor_attributions

def DeepLiftRects(image, model, data_classes, label):

    deepLiftFig, deepAtt = GenerateDeepLift(image, model, data_classes, label)

    print(deepLiftFig.shape)
    print(deepAtt.shape)

    ui = ChooseRectangles(deepAtt, [
        (200, 150, 100, 50),
        (0, 100, 200, 150),
        (100, 100, 200, 200),
    ])
    ui.draw()
    plt.show()
    print(ui.selected)

    return ui.selected


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
    print(dl_att)
    dl_att = np.transpose(dl_att.squeeze().cpu().detach().numpy(), (1, 2, 0))
    


    original_image = np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    # overlayed integrated gradients figure
    deepLiftFig,_ = viz.visualize_image_attr_multiple(dl_att, original_image, ["original_image","blended_heat_map"],
                                        ["all","all"],
                                        show_colorbar=True,
                                        titles=[f"Original(gt = {data_classes[label]}, predicted = {data_classes[predicted]})", "DeepLift"],
                                        fig_size=(10, 6))


    return deepLiftFig, dl_att

def GenerateBatchDeepLift(model, data_batch_iteration,data_classes,batch_it_size, save_file_dir):
    
    images, labels = data_batch_iteration.next()
    print('GroundTruth: ', ' '.join('%5s' % data_classes[labels[j]] for j in range(batch_it_size)))

    model.eval()
    # Get image classification
    model.to('cpu')
    print(images.shape)
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % data_classes[predicted[j]] for j in range(batch_it_size)))

    ind = 0

    for image in images:
        input = image.unsqueeze(0)
        input.requires_grad = True
        # Generate the visual explnations (saliency maps)
        # DeepLift
        deeplift = DeepLift(model)

        # Reset model's gradients
        model.zero_grad()
        dl_att = deeplift.attribute(input, target=labels[ind].item())
        dl_att = np.transpose(dl_att.squeeze().cpu().detach().numpy(), (1, 2, 0))

        original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        # overlayed integrated gradients figure
        deepLiftFig,_ = viz.visualize_image_attr_multiple(dl_att, original_image, ["original_image","blended_heat_map"],
                                            ["all","all"],
                                            show_colorbar=True,
                                            titles=[f"Original(gt = {data_classes[labels[ind]]}, predicted = {data_classes[predicted[ind]]})", "DeepLift"],
                                            fig_size=(10, 6))

        ind=ind+1
        #deepLiftFig.savefig(f"{save_file_dir}deepLiftB1_{ind}.png")

    
    return

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

def GenerateBatchExplanation(data_batch_iteration,batch_it_size, model, data_classes, save_file_dir):
    images, labels = data_batch_iteration.next()
    print('GroundTruth: ', ' '.join('%5s' % data_classes[labels[j]] for j in range(batch_it_size)))

    # Get image classification
    model.to('cpu')
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % data_classes[predicted[j]] for j in range(batch_it_size)))

    ind = 0
    for image in images:
        print(f"starting iteration {ind}")
        # Prepare the inputs to the the function that will generate the explanations
        input = image.unsqueeze(0)
        input.requires_grad = True
        # Generate the visual explnations (saliency maps)
        # Saliency Maps
        saliency = Saliency(model)
        # Reset model's gradients
        model.zero_grad()
        grads = saliency.attribute(input, target=labels[ind].item())
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        # Integrated Gradients
        ig = IntegratedGradients(model)
        # Reset model's gradients
        model.zero_grad()
        attr_ig, delta = attribute_image_features(ig, input, labels, ind,  baselines=input * 0, return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print('Approximation delta: ', abs(delta))
        print("Integrated Gradients computed")

        # Integrated Gradients w/ Noise Tunnel
        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)
        # Reset model's gradients
        model.zero_grad()
        attr_ig_nt = attribute_image_features(nt, input, labels, ind,  baselines=input * 0, nt_type='smoothgrad_sq', nt_samples=100, stdevs=0.2)
        attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        print("Integrated Gradients w/ noise computed")

        # Get occlusion maps
        occlusion = Occlusion(model)
        # Attribute occlusion features
        model.zero_grad()
        occ_att = occlusion.attribute(input, target=labels[ind].item(), sliding_window_shapes=(3, 15, 15))
        occ_att = np.transpose(occ_att.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print("Occlusion attribution computed")

        # And let's now visualise the different types of explanations
        print('Original Image')
        print('Predicted:', data_classes[predicted[ind]], ' Probability:', torch.max(F.softmax(outputs, 1)).item())
       
        original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

        # original image
        og_fig,_ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")
        
        # overlayed gradient magnitudes figure
        grad_mag_fig,_ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value", show_colorbar=True, title="Overlayed Gradient Magnitudes")

        # overlayed integrated gradients figure
        int_grad_fig,_ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")

        # Overlayed Integrated Gradients with SmoothGrad Squared figure
        int_smooth_grad_fig,_ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", outlier_perc=10, show_colorbar=True, title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

        # Occlusion based figre
        occlusion_fig,_ = viz.visualize_image_attr(occ_att, original_image, method="blended_heat_map", sign="positive", show_colorbar=True, title="Occlusion")

        og_fig.savefig(f"{save_file_dir}og_fig_{ind}.png")
        grad_mag_fig.savefig(f"{save_file_dir}grad_mag_fig_{ind}.png")
        int_grad_fig.savefig(f"{save_file_dir}int_grad_fig_{ind}.png")
        int_smooth_grad_fig.savefig(f"{save_file_dir}int_smooth_grad_fig_{ind}.png")
        occlusion_fig.savefig(f"{save_file_dir}occlusion_fig_{ind}.png")

        print(f"images saved to {save_file_dir}")
        print(f"Iteration {ind} complete")
        ind = ind+1

    return 


# Sort aux function: take third element for sort
def takeThird(elem):
    return elem[2]
