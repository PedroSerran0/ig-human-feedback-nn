import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import argsort
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def point_inside_rect(pt, rect):
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]


# 0123
# 4567
# 89...
#
#

# x = torch.rand(8, 8)
# ix = x.flatten().argsort()
# ix = ix[:5]
# line = ix // 8
# col = ix % 8
# rects = [((i//8)*h, (i%8)*w, ((i+1)//8)*h, ((i+1)%8)*w) for i in ix]


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
        plt.draw()

    def button_press(self, event):
        if event.button == 1 and event.inaxes:
            pt = event.xdata, event.ydata
            i = [i for i, rect in enumerate(self.rects) if point_inside_rect(pt, rect)]
            if len(i) > 0:
                i = i[0]
                self.selected.discard(i) if i in self.selected else self.selected.add(i)
                self.draw()

    def get_selected_rectangles(self):
        return {self.rects[i] for i in self.selected}



class GenerateRectangles:
    def __init__(self, img, size, stride, nr_rects):
        self.img = img
        self.size = size
        self.stride = stride
        self.nr_rects = nr_rects

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



def GetOracleFeedback(image, model_attributions, rectSize, rectStride, nr_rects):
    rectGenerator = GenerateRectangles(model_attributions, size=rectSize, stride=rectStride, nr_rects=nr_rects)
    rects = rectGenerator.get_ranked_patches()
    image = torch.permute(image,(1,2,0))

    ui = ChooseRectangles(image,rects)
    ui.draw()
    plt.show()
    print(ui.selected)

    return ui.selected

# myTensor = torch.rand(224,224)
# rectGenerator = GenerateRectangles(myTensor,size=14,stride=14)
# rects = rectGenerator.get_ranked_patches()


# if __name__ == '__main__':
#     from skimage.data import astronaut
#     ui = ChooseRectangles(myTensor,rects)
#     ui.draw()
#     plt.show()
#     print(ui.selected)

# if __name__ == '__main__':
#     from skimage.data import astronaut
#     ui = ChooseRectangles(astronaut(), [
#         (200, 350, 300, 450),
#         (0, 100, 300, 250),
#         (400, 100, 500, 400),
#     ])
#     ui.draw()
#     plt.show()
#     print(ui.selected)
