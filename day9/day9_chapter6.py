'''
day9: chapter6
'''
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet_err_backward:
    def __init__(self):
        pass
class TwoLayerNet_numerical:
    '''
    input size: size for input layer
    hidden_size: size for hidden layer
    output_size: size for output layer
    w: weight
    b: biasing
    '''
    def __init__(self, input_size, hidden_size,
                 output_size, weight_init_std = 0.01):
        self.params = {} #dictionary
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def softmax(self, x):
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    '''
    network computing
    '''
    def predict(self, x):
        w1, w2, b1, b2 = \
            self.params['w1'], self.params['w2'], self.params['b1'], self.params['b2']
        z1 = np.dot(x, w1)+b1
        z = self.sigmoid(z1)
        y1 = np.dot(z, w2)+b2
        y = self.softmax(y1)
        return y

    '''
    calculate f(x+h)-f(x-h)/(2*h)
    note:  h is the minimum value of delta
    '''
    def numerical_gradient(self, x):
        pass

    '''
    gradient: calculate by 'error backward propagation'
    x: input set
    t: test set
    '''
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        rdiff = 1
        rdiff = self.lastLayer.backward(rdiff)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            rdiff = layer.backward(rdiff)

        # set grads
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

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

