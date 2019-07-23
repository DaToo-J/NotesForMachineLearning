'''
1. 支持多个类型的layer：全连接、convolutional、max pooling、softmax
2. 支持多个激活函数：sigmoid、tanh、relu
3. 支持在GPU上跑
4. 加入了Theano

'''


import pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

# Activation functions for neurons
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)


# ----------------------------------------------------
# load mnist
def load_data_shared(filename="mnist.pkl.gz"):
    def shared(data):
        # 用 theano 处理mnist data 的格式，可以将data copy到GPU里
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    f = gzip.open(filename,'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return [shared(training_data), shared(validation_data), shared(test_data)]


class Network(object):
    def __init__(self, layers, mini_batch_size):
        # layers：描述网络结构，各个layer的type
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        # params：将各个layer type的参数都整合在一起
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)): # xrange() was renamed to range() in Python 3.
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_shared()
    mini_batch_size = 10
    net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    # net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)






