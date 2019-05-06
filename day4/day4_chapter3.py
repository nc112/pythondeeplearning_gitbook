import os, sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

#load_mnist
#return: (image,lable),(image,lable)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#Validation
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

#activate function
def softmax_original(x):
    return np.exp(x)/np.sum(np.exp(x))
def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

#validation:
test = np.array([1010,1000,990])
print(softmax_original(test))

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    print(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False)

#Validation
print(x_train)
print(t_train)
print(x_test)
print(t_test)

img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape)          # (784,)
img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
print(img.shape)          # (28, 28)

img_show(img)