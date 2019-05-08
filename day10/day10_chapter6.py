import numpy as np
from dataset.mnist import load_mnist

'''
return value:
(a,b), (c,d): 
a: train image
b: train label
C: test image
d: test label
'''
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

