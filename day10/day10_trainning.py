import numpy as np
from common.multi_layer_net import MultiLayerNet

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

'''
Define neural network parameters
'''
max_epochs = 200
train_size = 100
batch_size = 100




