import numpy as np
from common.multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

#Pre definition
ipsize = 784 #used for MNIST
hlist = [100, 100, 100, 100, 100, 100]
opsize = 10

class SGD:
    def __init__(self, lrate = 0.01):
        self.lrate = lrate
    '''
    parameter is a dictionary,
    grads is a result array, shall be calculated before call the function
    grads[] contain dl/dp
    '''
    def updatenetwork(self, parameters, grads):
        for key in parameters.keys():
            parameters -= self.lrate * grads[key]

#Initilize network
network = MultiLayerNet(input_size=ipsize, hidden_size_list=hlist, output_size=opsize)
optimizer = SGD()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
#Validation for over fitting
x_train = x_train[:300]
t_train = t_train[:300]


'''
Define neural network parameters
'''
max_epochs = 200
train_size = x_train.shape[0]
batch_size = 100
learning_step = 0

train_loss_list = []
train_accurate_list = []
test_accurate_list = []

'''
iter_per_epoch: complete one trainning
'''
iter_per_epoch = max(train_size/batch_size, 1)
epoch_count = 0

for i in range(learning_step):
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    '''
    Calculate grad:
    {
    x_batch: trainning set
    t_batch: test set
    }
    '''
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_accurate_list.append(train_acc)
        test_accurate_list.append(test_acc)

        epoch_count += 1
        if epoch_count >= max_epochs:
            break

class DrawFigure:
    def __init__(self):
        plt.title('Over fitting Chart')
        self.fig = plt.figure(1)
    def setXrange(self,xmin=None,xmax=None):
        if None == xmin or None == xmax:
            pass
        else:
            plt.xlim(xmin, xmax)
    def setYrange(self,ymin=None,ymax=None):
        if None == ymin or None == ymax:
            pass
        else:
            plt.ylim(ymin, ymax)
    def plotchart(self,x,y,position):
        self.fig.add_subplot(position[0], position[1], position[2])
        plt.plot(x,y)
    def plotshow(self):
        plt.show()

df = DrawFigure()
epoch =
df.plotchart(epoch_count, train_accurate_list, [1,2,1])
df.plotchart(epoch_count, test_accurate_list, [1,2,2])

