from keras.datasets.mnist import load_data
import numpy as np

# -----------------------------------------------   Preprocessing ---------------------------------------
# 0. loading datas
(train_digits, train_labels), (test_digits, test_labels) = load_data()
image_height = train_digits.shape[1]        # 28
image_width = train_digits.shape[2]         # 28
num_channels = 1            # grayscale images

# 1. reshape the images data to (num_samples, image_height, image_width, num_channels)
train_data = np.reshape(train_digits, (train_digits.shape[0], image_height, image_width, num_channels))
test_data = np.reshape(test_digits, (test_digits.shape[0],image_height, image_width, num_channels))
        # (60000, 28, 28, 1) &  (10000, 28, 28, 1)

# 2. re-scaling the images data to a values between (0.0, 1.0]
train_data = train_data.astype('float32') / 255.
test_data = test_data.astype('float32') / 255.


# 3. one-hot-encode the labels
# use "to_categorical()"
from keras.utils import to_categorical
num_classes = 10
train_labels_cat = to_categorical(train_labels,num_classes)         # (60000, 10)
test_labels_cat = to_categorical(test_labels,num_classes)           # (10000, 10)

# 4. split the training dataset
# 将training set 分为 train set 和 10% cross-validation set，在前者上做训练，在后者上做验证
# 不再test set上做训练，只用作最后的评估和预测。这样就能保证有些数据是模型没有见过的，防止过拟合。


# 4.1 shuffle the training dataset
for _ in range(5):
    indexes = np.random.permutation(len(train_data))
train_data = train_data[indexes]
train_labels_cat = train_labels_cat[indexes]

# 4.2 set the number of validation
val_perc = 0.10
val_count = int(val_perc * len(train_data))

# 4.3 split the training dataset into validation and training set
val_data = train_data[:val_count,:]
val_labels_cat = train_labels_cat[:val_count,:]
train_data2 = train_data[val_count:,:]
train_labels_cat2 = train_labels_cat[val_count:,:]



# -----------------------------------------------   Building CNN ---------------------------------------
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(image_height, image_width, num_channels)))
        # filters ：滤波器的数量
        # kernel_size ： 2D卷积窗口的宽度和高度
        # input_shape ：使用 con2D 作为第一层时，需要提供 input_shape tuple
        # output：(None, 28, 28, 32)， 依然为 (28,28) 是因为 padding=‘same’
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))        # output：(None, 3, 3, 64)
    # model.add(Dropout(0.25))              # 也可以 Dropout 一下
    model.add(Flatten())
        # output：(None, 576)；
        # Flatten()：将输入“压平”，多维输入一维化。常用在卷积层到全连接层的过度

    model.add(Dense(128, activation='relu'))
        # output：(None, 128)；
        # Dense()：全连接层，按照activation逐个元素计算激活值。如果全连接为第一层，则需要设置 input_shape
        # 全连接： 上一层的每一个神经元，都和下一层的每一个神经元是相互连接的
        # 参数个数： (上一层神经元个数 + 1) * 全连接层神经元个数

    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
print(model.summary())


# -----------------------------------------------   Training model ---------------------------------------
results = model.fit(train_data2, train_labels_cat2, epochs=10, batch_size=64, validation_data=(val_data, val_labels_cat))
    # fit()：会返回一个 ‘history’字典，keys() = ['acc','loss', 'val_acc', 'val_loss']，values() is a list with epochs elements
model.save('save/my_cnn_model.h5')


# -----------------------------------------------   Evaluating model ---------------------------------------
# from keras.models import load_model
# model = load_model('save/my_cnn_model.h5')
test_loss, test_accuracy = model.evaluate(test_data, test_labels_cat, batch_size=64)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))


# -----------------------------------------------   Prediction  ---------------------------------------
predictions = model.predict(test_data)
first20_preds = np.argmax(predictions, axis=1)[:25]
first20_true = np.argmax(test_labels_cat,axis=1)[:25]
print(first20_preds)
print(first20_true)

