import numpy as np
class MyTrainer:
    def __init__(self, networktype, trainning_loop, train_size, batch_size, x_train, t_train, optimizer='SGD'):
        self.networktype = networktype
        self.trainning_loop = trainning_loop
        self.train_size = train_size
        self.batch_size = batch_size
        self.x_train = x_train
        self.t_train = t_train
        self.optimizer = optimizer
    def train(self):
        for i in range(trainning_loop):
            batch_mask = np.random.choice(train_size, batch_size)

            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            '''
            Calculate grad:
            {
            x_batch: trainning set
            t_batch: test set
            }
            '''
            grads = networktype.gradient(x_batch, t_batch)
            if optimizer == 'SGD':
                optimizer.update_network(networktype.params, grads)
            else:
                raise Exception

            iter_per_epoch = max(self.train_size / mini_batch_size, 1)
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_accurate_list.append(train_acc)
                test_accurate_list.append(test_acc)

                epoch_count += 1
                if epoch_count >= max_epochs:
                    break