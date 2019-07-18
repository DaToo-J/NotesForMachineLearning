'''
1. 使用随机梯度下降学习算法
2. 添加了一些：交叉熵代价函数、正则化、更好地w初始化
'''
import json
import random
import sys
import numpy as np

# -------------------------------------------------------------------
# 其他func
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # 对 sigmoid 求导
    return sigmoid(z)*(1-sigmoid(z))


def vectorized_result(j):
    # 输入数字j，返回其 onehot 数组
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

# -------------------------------------------------------------------
# cost functions
class QuadraticCost():
    @staticmethod
    def fn(a,y):
        # 二范数
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost():
    @staticmethod
    def fn(a,y):
        # np.nan_to_num：将nan转为0.0
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

# -------------------------------------------------------------------
# network
class Network():
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        # 初始化w：高斯分布，均值为0，标准差 / 根号下w
        # 初始化b：高斯分布，均值为0，标准差为1
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,                  # 正则化参数
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        # return：tuple(每个epoch在验证集上的代价， 每个epoch在验证集上的准确率， 训练集的代价， 训练集的准确率)
        # monitor：用于monitor测试集、验证集的准确率，如未设置，则不monitor，返回的tuple里第一个元素为0。否则，为epoch size大小的list

        best_accuracy = 1
        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_acc = [], []
        training_cost, training_acc = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            # ---------------------------------------------------------

            if monitor_evaluation_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_acc.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_acc.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # ---------------------------------------------------------
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                else:
                    no_accuracy_change += 1
                if (no_accuracy_change == early_stopping_n):
                    return evaluation_cost, evaluation_acc, training_cost, training_acc

        return evaluation_cost, evaluation_acc, training_cost, training_acc

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        '''
        :param mini_batch: list of tuples(x,y)
        :param eta: learning rate
        :param lmbda: regularization parameter
        :param n: total size of the training data set
        :return:
        '''


        # 全零初始化nabla_b、nabla_w，最后得到的是这个mini-batch的b、w，其shape和整体的b、w一样，此后还要和整体的b、w用学习率更新
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # ????
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        activation = x
        activations = [x]



    def total_cost(self, data, lmbda, convert=False):
        # 返回 total cost
        # convert的设置和 accuracy() 相反
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert:
                # 把 验证集、测试集都onehot一下下
                y = vectorized_result(y)
            #     ？？？
            cost += self.cost.fn(a,y) / len(data)
            cost += 0.5 * (lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost

    def accuracy(self, data, convert=False):
        # 对应 network.py 的 evaluate()
        # 返回正确outputs的个数
        # convert false：如果是验证集或测试集
        # convert true：如果是训练集
        # 因为它们的y的形式不同，mnist_loader.load_data_wrapper，这样电脑计算会快一点
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)),y) for (x,y) in data]
        result_accuracy = sum(int(x==y) for (x,y) in results)
        return result_accuracy

