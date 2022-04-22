import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    def __init__(self, img, rects, edgecolor='blue'):
        self.img = img
        self.rects = rects
        self.selected = set()
        self.edgecolor = edgecolor
    
    def draw(self):
        plt.imshow(self.img)
        for i, (x1, y1, x2, y2) in enumerate(self.rects):
            edgecolor = 'green' if i in self.selected else self.edgecolor
            plt.gca().add_patch(Rectangle(
                (x1, y1), x2-x1, y2-y1,
                facecolor='none', edgecolor=edgecolor, lw=3))
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

if __name__ == '__main__':
    from skimage.data import astronaut
    ui = ChooseRectangles(astronaut(), [
        (200, 350, 300, 450),
        (0, 100, 300, 250),
        (400, 100, 500, 400),
    ])
    ui.draw()
    plt.show()
    print(ui.selected)
