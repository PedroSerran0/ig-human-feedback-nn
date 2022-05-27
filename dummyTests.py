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
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import pandas as pd


myTensor = [[ 0.1126,  0.0123,  0.0713,  0.1200,  0.0913],
        [ 0.1303, -0.0309,  0.0744,  0.1433,  0.0611],
        [ 0.1058, -0.0396,  0.0520,  0.0883,  0.0643],
        [ 0.1165, -0.0747,  0.0559,  0.1075,  0.0983],
        [ 0.1260, -0.0623,  0.1107,  0.1495,  0.1112],
        [ 0.1036, -0.0376,  0.0323,  0.1091,  0.0437],
        [ 0.1149, -0.0340,  0.1628,  0.1864,  0.0314],
        [ 0.0905,  0.0026,  0.0508,  0.1045,  0.0558],
        [ 0.1283, -0.0172,  0.0733,  0.1478,  0.1399],
        [ 0.0524, -0.0238,  0.0151,  0.0926,  0.0528]]
myTensor = torch.FloatTensor(myTensor)
myTensor = torch.add(myTensor, 0.0001)

epochs = [0,1,2,3,4,5,6,7,8,9,10]
x1 = [20,23,25,26,27,28,29,35,67,32]
x2 = [0.1,0.2,0.5,0.7,0.9,0.7,0.9,1.1,1.5,1.7]
x3 = [200,203,250,260,270,280,290,305,670,320]

plt.figure(figsize=(10,5))
plt.title(f"Training and Validation Loss (_AL)")
plt.plot(x1,label="train-loss")
plt.plot(x2,label="val-loss")
plt.plot(x3,label="train-loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
#plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_loss_{EPOCHS}E_AL.png"))
plt.show()

transf = transforms.ToPILImage()
img1 = transf(myTensor)
img1 = img1.save("/home/up201605633/Desktop/tensor2.jpg")

probs = torch.softmax(myTensor,1)

myList = list()

a = ["a","a",1.2]
b=["b","a",1.6]
c=["c","a",0.9]
d=["d","a",1.4]
myList.append(a)
myList.append(b)
myList.append(c)
myList.append(d)
#print(myList)
#print(myList[0][2])

def getMaxEntropyImage(data_list):
        for data_point in data_list:
                maxvalue = 0
                for i in range(len(data_list)):
                        entNum = data_list[i][2]
                        if(entNum > maxvalue):
                                maxvalue = entNum
                                maxEntropyImage = data_list[i]
        return maxEntropyImage 

# take second element for sort
def takeInt(elem):
    return elem[2]

myList.sort(key=takeInt, reverse=True)
print(myList[1][2])

# df = pd.DataFrame(myList, columns = ['Column_A','Column_B','Column_C'])

# print(df)

# plt.hist(df['Column_C'], color = 'blue', edgecolor = 'black',
#          bins = int(5))
# plt.savefig("attempt1")