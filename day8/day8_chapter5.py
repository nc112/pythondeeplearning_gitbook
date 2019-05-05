import numpy as np
import matplotlib.pyplot as plt


class AddLayer:
    def __init__(self):
        pass
    def forwardflow(self, a, b):
        c = a + b
        return c
    def backwardflow(self, rdiff):
        ra = rdiff * 1
        rb = rdiff * 1
        return ra, rb

class MulLayer:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
    def forwardflow(self, a, b):
        c = a * b
        self.a = a
        self.b = b
    def backwardflow(self, rdiff):
        ra = rdiff * self.b
        rb = rdiff * self.a
        return ra, rb

#sigmoid
def sigmoidfunc(x):
    return 1/(1+np.exp(-x))

class Sigmoid:
    def __init__(self):
        self.result = None
    def forward(self, x):
        result = sigmoidfunc(x)
        self.result = result
        return result

    def backward(self, rdiff):
        ra = rdiff * (1.0 - self.result) * self.result
        return ra

#ReLu
def ReLUfunc(x):
    return np.max(0, x)
class ReLu:
    def __init__(self):
        self.radiff = None
    def forwardflow(self, x):
        result = ReLUfunc(x)
        for item in result:
            if item >= 0:
                self.radiff[] = 1
            else:
                self.radiff = 0
        return result
    def backwardflow(self, rdiff):
        return rdiff * self.radiff