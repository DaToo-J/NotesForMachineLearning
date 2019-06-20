# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader


def main(_):
    # 1. 定义 x 和 y
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 100) + x / 200. + noise
    plt.plot(x, y)
    plt.savefig('2_timeseries_y.jpg')

    # 2. 生成 x 和 y 的data字典
    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
    }

    # 3. 创建reader，生成batch数据
    reader = NumpyReader(data)
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=16, window_size=40)       # 每个batch中有16个序列，每个序列长度为40

    # 4. 定义 AR 模型（Autoaggressive model），是统计学上处理时间序列模型的基本方法
    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=200,          # 序列的规律性周期
        input_window_size=30,       # 输入的值
        output_window_size=10,      # 输出的值，其中，window_size = input_window_size + output_window_size
        num_features=1,             # 在某个时间点上观察到的值的维度
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS
                                    # 损失，有2种可选， NORMAL_LIKELIHOOD_LOSS , SQUARED_LOSS
    )

    # 5. 训练 training
    ar.train(input_fn=train_input_fn, steps=6000)

    # 6. 验证 evaluation ---- 使用训练好的模型在原先的训练集上进行计算，观察模型的拟合效果
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
            # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
            # mean ： 保存970个预测值； times： 前者对应的时间点； loss ：总的损失；
            # start_tuple：用于之后的预测，保存最后30步的输出值和时间点

    # 7. 预测 prediction
    (predictions,) = tuple(ar.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(evaluation, steps=250)))
            # 在1000步之后，向后预测了250个时间点，对应的值保存在 prediction['mean']

    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('predict_result.jpg')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()