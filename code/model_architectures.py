# PyTorch Imports
import torch
import torch.nn as nn
import torchvision.models as models



# Define backbone models
resnet50 = models.resnet50(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
efficientnet_b1 = models.efficientnet_b1(pretrained=True)



# Class: PretrainedModel
class PretrainedModel(torch.nn.Module):
    def __init__(self, pretrained_model, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs
        model = getattr(models, pretrained_model)(pretrained=True)
        
        # remove last layer from pre-trained model 
        model = nn.Sequential(*tuple(model.children())[:-1])
        
        # get last dimension of the model
        last_dimension = torch.flatten(model(torch.randn(1, 3, 224, 224))).shape[0]
        self.model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(last_dimension, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        )


    def forward(self, x):
        return self.model(x)
