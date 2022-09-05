# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# PyTorch Imports
import torch
import torch.nn as nn



# Function: See if a point is inside a rectangle
def point_inside_rect(pt, rect):
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]



# Class: Choose rectangles (UI based in Matplotlib)
class ChooseRectangles:
    def __init__(self, img, rects, edgecolor='red'):
        self.img = img
        self.rects = rects
        self.selected = set()
        self.edgecolor = edgecolor
    
    def draw(self):
        plt.clf()
        plt.imshow(self.img)
        for i, (x1, y1, x2, y2) in enumerate(self.rects):
            if i in self.selected:
                plt.gca().add_patch(Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    facecolor=self.edgecolor, edgecolor='none', alpha=0.4))
            plt.gca().add_patch(Rectangle(
                (x1, y1), x2-x1, y2-y1,
                facecolor='none', edgecolor=self.edgecolor, lw=3))
        plt.connect('button_press_event', self.button_press)
        plt.connect('key_press_event', self.key_press)
        plt.draw()


    # Method: Keyboard interface
    def key_press(self, event):
        if event.key in ['enter', 'escape']:
            plt.close()


    # Method: Button interface
    def button_press(self, event):
        if event.button == 1 and event.inaxes:
            pt = event.xdata, event.ydata
            i = [i for i, rect in enumerate(self.rects) if point_inside_rect(pt, rect)]
            if len(i) > 0:
                i = i[0]
                self.selected.discard(i) if i in self.selected else self.selected.add(i)
                self.draw()


    # Method: Get selected rectangles
    def get_selected_rectangles(self):
        return {self.rects[i] for i in self.selected}



# Class: Generate rectangles (UI based on Matplotlib)
class GenerateRectangles:
    def __init__(self, img, size, stride, nr_rects):
        self.img = img
        self.size = size
        self.stride = stride
        self.nr_rects = nr_rects


    # Method: Get ranked patches
    def get_ranked_patches(self):
        avg = nn.AvgPool2d(self.size, self.size)
        avg_patches = avg(self.img[None])

        print('avg_patches:', avg_patches.shape)

        # Sort patches
        ranks = torch.argsort(avg_patches.flatten()).reshape((avg_patches.shape))
        print('ranks shape:', ranks.shape)
        
        _, yi, xi = torch.where(ranks < self.nr_rects)


        rects = [(x*self.size, y*self.size, (x+1)*self.size, (y+1)*self.size) for y, x in zip(yi, xi)]

        return rects



# Function: Image show function
def imshow(img ,transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()



# Function: Get Oracle Feedback (UI)
def GetOracleFeedback(image, label, idx, model_attributions, pred, rectSize, rectStride, nr_rects):
    rectGenerator = GenerateRectangles(model_attributions, size=rectSize, stride=rectStride, nr_rects=nr_rects)
    rects = rectGenerator.get_ranked_patches()
    image = image / 2 + 0.5     # unnormalize
    #npimg = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    
    ui = ChooseRectangles(image,rects)
    ui.draw()
    plt.title(f"True Label: {label}  Prediction: {pred}  Image idx: {idx}")
    plt.show()
    print(ui.selected)
    selected_rects = ui.get_selected_rectangles()

    return ui.selected, selected_rects
