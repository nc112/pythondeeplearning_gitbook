import numpy as np
from common.multi_layer_net import MultiLayerNet

#Pre definition
ipsize = 784 #used for MNIST
hlist = [100, 100, 100, 100, 100, 100]
opsize = 10

#
network = MultiLayerNet(input_size=ipsize, hidden_size_list=hlist, output_size=opsize)