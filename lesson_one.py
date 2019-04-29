import numpy as np
import matplotlib.pyplot as plt

class DateGenerate:
    def __init__(self):
        self.step = 0.1
        self.startnumber = 0
        self.stopnumber = 6
    def generateSin(self, start=None, stop=None, step=None):
        x=[]
        y=[]
        if None == start or\
            None == stop or\
            None == step:
            x = np.arange(self.startnumber, self.stopnumber, self.step)
            y = np.sin(x)
        else:
            x = np.arange(start, stop, step)
        return x,y

class PlotArray:
    def __init__(self):
        pass
    def PlotandShow(self, x=None, y=None):
        plt.plot(x, y)
        plt.show()

dg = DateGenerate()
dginput = dg.generateSin()

pa = PlotArray()
pa.PlotandShow(dginput[0], dginput[1])