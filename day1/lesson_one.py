import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

class DateGenerate:
    def __init__(self):
        self.step = 0.1
        self.startnumber = 0
        self.stopnumber = 6
    def generateSin(self, start=None, stop=None, step=None):
        x = []
        ysin = []
        ycos = []
        if None == start or\
            None == stop or\
            None == step:
            x = np.arange(self.startnumber, self.stopnumber, self.step)
            ysin = np.sin(x)
            ycos = np.cos(x)
        else:
            x = np.arange(start, stop, step)
        return x,ysin,ycos

class PlotArray:
    def __init__(self):
        pass
    def PlotChart(self, x=None, ysin=None, ycos=None):
        #Title
        plt.title('Sin&Cos Chart')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.plot(x, ysin, linestyle='--', label='Sin')
        plt.plot(x, ycos, label='Cos')
        plt.legend()

dg = DateGenerate()
dginput = dg.generateSin()

pa = PlotArray()
#pa.PlotChart(dginput[0], dginput[1], dginput[2])

#input png image
img = imread('lena.png')
plt.imshow(img)
plt.show()