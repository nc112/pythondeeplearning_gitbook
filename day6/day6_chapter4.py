import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#define a simple function:
#y=0.01x^2+0.1x
def function_test(x):
    return 0.01*x**2 + 0.1*x

#define numerical diff function
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(h*2)

#validation
x = np.arange(0.0, 20.0, 0.1)
y = function_test(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x,y)
#plt.show()

# Integration of mini-batch
(x_train, t_train), (x_test, t_test) =  load_mnist(normalize=True,
                                                   one_hot_label = True)


#human setting parameters
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)
sizeof_hidden_elements = 2

network = TwoLayerNet(input_size=784, hidden_size=sizeof_hidden_elements, output_size=10)

for i in range(iters_num):
    # mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch) #high speed version

    # learning
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))