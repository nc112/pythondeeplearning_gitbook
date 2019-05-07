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

#Validation for over fitting
x_train = x_train[:300]
t_train = t_train[:300]