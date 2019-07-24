from __future__ import print_function


import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

'''
在[0,4] 上训练一下简单的 convnet，再fine-tune dense layers 来分类 [5,9]
'''

now = datetime.datetime.now

# -----------------------------------------------   Preprocessing ---------------------------------------
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# create two datasets one with digits below 5 and one with 5 and above
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

# 将 [5,9] 的数字映射到 [0,4]
x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5


# -----------------------------------------------  Model ---------------------------------------
def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)      # (30596, 28, 28, 1) training
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)         # (5139, 28, 28, 1) training
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,                            # 日志显示模式； 0=安静模式， 1=进度条， 2=每轮一行
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# -----------------------------------------------   Parameters ---------------------------------------
batch_size = 128
num_classes = 5
epochs = 5


img_rows, img_cols = 28, 28                         # input image dimensions
filters = 32                                        # number of convolutional filters to use
pool_size = 2                                       # size of pooling area for max pooling
kernel_size = 3                                     # convolution kernel size

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
    # (28,28,1)



# -----------------------------------------------  2 Layers Lists ---------------------------------------
# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# ---------------------------------------------- Training model [0,4]---------------------------------------
# create complete model
model = Sequential(feature_layers + classification_layers)

# train model for 5-digit classification [0..4]
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

model_json = model.to_json()
with open("save/cnn_transfer_1.json", "w", encoding='utf-8') as json_file:
    json_file.write(model_json)


# ---------------------------------------------- Training model [5,9]---------------------------------------
# freeze feature layers and rebuild model
# freeze：意味着 weights 不会更新。通过设置 'trainable' 为 True or False
# 设置完 ‘trainable’ 属性之后，需要 compile() 一下
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)



model_json = model.to_json()
with open("save/cnn_transfer_2.json", "w", encoding='utf-8') as json_file:
    json_file.write(model_json)





