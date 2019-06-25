import numpy as np
import os
import scipy.io as sio


def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([x.shape[0], -1]) / 255.0
    print('MNIST:', x.shape)
    return x, y


def load_mnist_test():
    from tensorflow.keras.datasets import mnist
    _, (x, y) = mnist.load_data()
    x = x.reshape([x.shape[0], -1]) / 255.0
    print('MNIST-TEST:', x.shape)
    return x, y


def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([x.shape[0], -1]) / 255.0
    print('Fashion MNIST:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16*16])
    print('USPS samples', x.shape)
    return x, y


def load_data(dataset):
    dataset = dataset.lower()
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps()



