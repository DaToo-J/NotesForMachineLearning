# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

'''
时间序列
1. 输入数据的读取：可从numpy生成，可从csv读取
    - numpy：
        + 定义 输入x，输出y
        + 创建reader
        + 可从reader中读取 full data， 也可从reader生成 batch数据
          但是，都得 需要 start_queue_runners 启动队列才可以用 run()获取值
          
    -  csv：
        + 将前3步换为：
            csv_file_name = './data/period_trend.csv'
            reader = tf.contrib.timeseries.CSVReader(csv_file_name)
'''

# 1. 定义 观察的时间点x 和 观察到的值y
#    其中，y是sin加上噪声后的值
x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise
plt.plot(x, y)
plt.savefig('1_timeseries_y.jpg')

# 2. 将 x 和 y 保存到data字典里
data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
}

# 3. 利用 NumpyReader 将其读取为一个 reader
reader = NumpyReader(data)

# 4. 真正读取 reader 的值
with tf.Session() as sess:
    full_data = reader.read_full()      # 用read_full 读取 reader 得到一个时间序列对应的 tensor， 需要 start_queue_runners 启动队列才可以用 run()获取值
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(full_data))
    coord.request_stop()

# 5. 建立读取batch数据的对象
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=2, window_size=10)        # 一个batch里有2个序列，每个序列长度为10

# 6. 真正读取、create batch数据
with tf.Session() as sess:
    batch_data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    one_batch = sess.run(batch_data[0])
    print('one_batch_data:', one_batch)
    coord.request_stop()

