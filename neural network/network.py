import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # 对 sigmoid 求导
    return sigmoid(z)*(1-sigmoid(z))

class Network():
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes

        # 第i层b-matrix = [第i层神经元个数, 1]
        # self.biases = list(all b-matrix)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # 第i层w-matrix = [第i层神经元个数, 第i-1层神经元个数]
        # self.weights = list(all w-matrix)
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    # 2. 根据w、b计算每一层的a
    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    # 3. 传递到最后一层，计算代价 (output - y)
    def cost(self, output, y):
        return (output - y)

    # 4. 反向传播，计算每一层的dw、db，存入nabla_w、nabla_b
    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for b,w in zip(self.biases, self.weights):
            # 计算每一层的z，并存入zs
            z = np.dot(w, activation) + b
            zs.append(z)
            # 计算每一次的激活值activation，并存入activations
            # activations = [x, a1, a2, ..., al]
            # activations：第一个元素是x，是因为dw 是由上一层的a决定，所以在计算 dw1时，需要a0
            activation = sigmoid(z)
            activations.append(activation)

        # 根据最后一层的激活值计算该层的 db 和 dw
        # delta -- dC/dz = da/dz * dC/da = sigmoid'(z) *  2(a - y)
        delta = self.cost(activations[-1], y) * sigmoid_prime(zs[-1])
        # nabla_b -- dC/db = dz/db * dC/dz = 1 * delta
        nabla_b[-1] = delta
        # nabla_w -- dC/dw = dz/dw * dC/dz = a^(L-1) * delta
        # dC/dw：是由上一层的激活值决定
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 于是接着从倒数第二层遍历，计算 db 和 dw，并保存
        # 所以 l 是从2开始range，下标是 -l
        for l in range(2, self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)       # 求z的导数
            # 每一层的delta都可以由下一层的delta迭代计算出来
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # 用 mini-batch 来更新 w、b，eta为学习率
    def update_mini_batch(self, mini_batch, eta):
        # 全零初始化nabla_b、nabla_w，最后得到的是这个mini-batch的b、w，其shape和整体的b、w一样，此后还要和整体的b、w用学习率更新
        # mini-batch = （x,y）
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 用mini-batch的b、w和整体的b、w更新
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

    # 随机梯度下降
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # training_data is a list of tuples (x,y)
        # 如果有test_data，则在每个epoch之后，都evaluate一次test_data，但是会慢一些
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            # 将training_data 拆分为 mini_batch_size 个 mini_batch， 存入mini_batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 遍历各个 mini_batch，更新self.weights、self.biases
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete.".format(j))

    def evaluate(self, test_data):
        # 把 test_data 喂给 feedforward，返回各层的 a值 (此时用的是已经更新过的self.weights、self.biases)
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

if __name__ == "__main__":
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    sizes = [784, 30, 10]
    nw = Network(sizes)
    nw.SGD(training_data,30,10,3.0, test_data=test_data)
    # print(sizes[1:])
    # print(sizes[:-1])
    # print(nw.biases)
    # print(nw.weights)
    # # print([(y,x) for x,y in zip(sizes[:-1], sizes[1:])])
    # print(np.array([1,2,3,4]))