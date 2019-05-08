import numpy as np
from common.multi_layer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

class DrawFigure:
    def __init__(self, title):
        plt.title(title)
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
    def plotchart(self,x,y):
        plt.plot(x,y)
    def plotshow(self):
        plt.show()


def drawchart(title, ylist):
    df = DrawFigure(title)
    epoch = np.arange(0, epoch_count, 1)
    for item in ylist:
        df.plotchart(epoch, item)
    df.plotshow()


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    '''
    * convert paras to tuple
    ** convert params to dictionary
    '''
    def forward(self, x, train_flg=True):
        if train_flg:

            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class SGD:
    def __init__(self, lrate=0.01):
        self.lrate = lrate

    '''
    parameter is a dictionary,
    grads is a result array, shall be calculated before call the function
    grads[] contain dl/dp
    '''

    def update_network(self, parameters, grads):
        for key in parameters.keys():
            parameters[key] -= self.lrate * grads[key]

# Pre definition
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
#Validation for over fitting
x_train = x_train[:300]
t_train = t_train[:300]

ipsize = 784  # used for MNIST
hlist = [100, 100, 100, 100, 100, 100]
opsize = 10
use_dropoutFlag = True
dp_ration = 0.15

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100
#trainning_loop = int(1e9)

train_loss_list = []
train_accurate_list = []
test_accurate_list = []

'''
#Define neural network parameters
{
    max_epochs: trainning number
    train_size: size of the train data
    batch_size: size of the batch
}
iter_per_epoch: complete one trainning
'''
iter_per_epoch = max(train_size/batch_size, 1)
epoch_count = 301

#Initilize network
#network = MultiLayerNet(input_size=ipsize, hidden_size_list=hlist, output_size=opsize)
network = MultiLayerNetExtend(input_size=ipsize, hidden_size_list=hlist, output_size=opsize,
                              use_dropout=use_dropoutFlag, dropout_ration=dp_ration)
optimizer = SGD()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=epoch_count, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()
train_accurate_list, test_accurate_list = trainer.train_acc_list, trainer.test_acc_list

#Valitation
drawchart('Optimize network by Dropout', [train_accurate_list, test_accurate_list])