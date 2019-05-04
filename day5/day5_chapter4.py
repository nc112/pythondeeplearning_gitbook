import numpy as np
from dataset.mnist import load_mnist

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#loss function
'''
mean squared error:
##E=\frac{1}{2}\sum_k(y_k-t_k)^2
'''
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

'''
##E=-\frac{1}{N}\sum_n\sum_k t_{nk}\log y_{nk}
'''
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size())
        y = y.reshape(1, y.size())
    batch_size = y.shape[0]
    return -np.sum(np.sum(t * np.log(y+1e-7))) / batch_size

#validation
t=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(y1,t))

y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(mean_squared_error(y2,t))

#validation
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)