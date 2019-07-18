import pickle
import gzip
import numpy as np

def load_data():
    # 读取 mnist数据集，不做任何处理
    # training data：tuple（x， y）
    #             x：ndarray(50000 * 784)
    #             y：ndarray(50000 * 1)
    # validation/test data：都只有10000条
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data )

def load_data_wrapper():
    # 将读取的mnist数据集，调整一下格式，方便输入到nn里

    # 1. 加载数据集
    tr_d, va_d, te_d = load_data()

    # 2. 调整training data 里的 x、y的格式，好输入到neural network里
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
        # tr_d[0]：是所有training data的x，(50000, 784)
        # x.shape = (784,)
    training_results = [vectorized_result(y) for y in tr_d[1]]
        # tr_d[1]：是所有training data的y，(50000, 1)
        # 给 y 进行 onehot 编码
    training_data = zip(training_inputs, training_results)

    # 3. validation data 和 test data 只用调整x的格式
    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_data = zip(validation_inputs, te_d[1])

    return (training_data, validation_data, test_data )

def vectorized_result(j):
    # 输入数字j，返回其 onehot 数组
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    pass
