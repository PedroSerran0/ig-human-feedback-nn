from torchvision.datasets import SVHN
from torch.utils import data
from torchvision.models import vgg16
from torchvision import transforms
import torch

tr = SVHN('data', download=True, transform=transforms.ToTensor())
tr = data.Subset(tr, range(0, 1000))  # undersample to speed up
tr = data.DataLoader(tr, 256, True)

model = vgg16()
model = model.cuda()

opt = torch.optim.Adam(model.parameters())
ce = torch.nn.CrossEntropyLoss()
model.train()

def my_loss(Ypred, X):
    # it "works" with both retain_graph=True and create_graph=True, but I think
    # the latter is what we want
    grad = torch.autograd.grad(Ypred.sum(), X, create_graph=True)[0]
    # grad has a shape: torch.Size([256, 3, 32, 32])
    return (grad**2).mean()

for epoch in range(10):
    print(f'* Epoch {epoch+1}')
    avg_loss = 0
    for X, Y in tr:
        opt.zero_grad()
        X = X.cuda()
        Y = Y.cuda()
        X.requires_grad = True  # <-- necessary before calling the model
        Ypred = model(X)
        loss = ce(Ypred, Y) + my_loss(Ypred, X)
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
    print('Loss:', avg_loss)