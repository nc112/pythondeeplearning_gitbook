'''
chapter3
'''
import numpy as np
import matplotlib.pylab as plb
#sigmoid function
def sigmoid(x):
    return  1/(1 + np.exp(-x))

#Step function
def stepfunc(x):
    y = x > 0
    return y.astype(np.int)

def ReLUfunc(x):
    return np.maximum(0, x)

class DrawFigure:
    def __init__(self):
        self.yrange = [-0.1, 1,1]
        plb.title('sigmod/step/ReLU Chart')
        plb.ylim(self.yrange[0], self.yrange[1])
        self.fig = plb.figure(1)
    def setXrange(self,xmin,xmax):
        plb.xlim(xmin, xmax)
    def setYrange(self,ymin,ymax):
        plb.ylim(ymin, ymax)
    def plotchart(self,x,y,position):
        self.fig.add_subplot(position[0], position[1], position[2])
        plb.plot(x,y)
    def plotshow(self):
        plb.show()

#validation
x = np.array([1,2])
w = np.array([[1,3,5], [2,4,6]])
y = np.dot(x,w)
print(y)
y = ReLUfunc(x)
print(y)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
df = DrawFigure()
df.plotchart(x,y, [1,3,1])
y = stepfunc(x)
df.plotchart(x,y, [1,3,2])
y = ReLUfunc(x)
df.plotchart(x,y, [1,3,3])

df.plotshow()