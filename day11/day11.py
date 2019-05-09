import numpy as np
import sys, os

sys.path.append(os.pardir)
from common.util import im2col

#convolution
x1 = np.array([[1,2,3,0],[0,1,2,3],[3,0,1,2],[2,3,0,1]]).flatten()
w = np.array([[2,0,1],[0,1,2],[1,0,2]]).flatten()
#w = w.flatten()
print(x1.shape, '\r\n', w.shape)

print('\r\n',np.convolve(x1, w, 'valid'))

#validaton