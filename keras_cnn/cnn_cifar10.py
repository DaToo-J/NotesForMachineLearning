from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

from keras.datasets.cifar10 import load_data
import numpy as np

# -----------------------------------------------   Parameters ---------------------------------------
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20


# -----------------------------------------------   Preprocessing ---------------------------------------
(x_train, y_train), (x_test, y_test) = load_data()      # train :  (50000, 32, 32, 3)
                                                        # test : (10000, 32, 32, 3)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# -----------------------------------------------  Model ---------------------------------------
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

model = build_model()
print(model.summary())

# ------------------------------------------ Training Model ---------------------------------------

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,   # 数据集去中心化，使其均值为0
        samplewise_center=False,    # 使输入数据的每个样本均值为0，set each sample mean to 0
        featurewise_std_normalization=False,    # 每个输入样本都除以数据集的标准差，divide inputs by std of the dataset
        samplewise_std_normalization=False,     # 每个输入样本都除以它的标准差， divide each input by its std
                                                # featurewise：从整个数据集分布去考虑，samplewise：只针对图片本身
        zca_whitening=False,                    # zca白化：针对图片进行pca降维，减少冗余信息，保留重要特征。apply ZCA whitening
        zca_epsilon=1e-06,                      # epsilon for ZCA whitening
        rotation_range=0,                       # 在用户指定旋转角度范围内随机旋转，randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,                  # 水平平移，平移距离在 [0, 最大平移距离] 内，最大平移距离 = 图片长 * 参数
                                                # 一般会超出原图范围的区域会根据 “fill_mode”参数来补全
                                                # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,
                                                # randomly shift images vertically (fraction of total height)
        shear_range=0.,                         # 错切变换，让x坐标保持不变，对应的y坐标按比例发生平移。比如，把矩形拉成平行四边形。set range for random shear
        zoom_range=0.,                          # 对长或宽等比缩放。可为一个int，可为一个list。set range for random zoom
        channel_shift_range=0.,                 #对颜色通道的数值进行偏移，来改变图片颜色。 set range for random channel shifts
                                                # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,                                # value used for fill_mode = "constant"
        horizontal_flip=True,                   # 对图片进行随机水平翻转。randomly flip images
        vertical_flip=False,                    # randomly flip images
        rescale=None,                           # set rescaling factor (applied before any other transformation)
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0],
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# ------------------------------------------ Save and Score ---------------------------------------

save_dir = os.path.join(os.getcwd(), 'save')
model_name = 'keras_cifar10_model_datagen.h5'
weight_name = 'keras_cifar10_weights_datagen.h5'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
weight_path = os.path.join(save_dir, weight_name)
model.save(model_path)
model.sample_weights(weight_path)
print('Saved trained model at %s ' % model_path)
print('Saved weights  at %s ' % weight_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])