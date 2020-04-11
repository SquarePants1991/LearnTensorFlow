import tensorflow as tf
from keras.utils import np_utils
import numpy as np
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
import keras.backend as kbackend
from keras import losses, metrics, optimizers, models

import matplotlib.pyplot as plt

def show_data(imgs, labels):
    plt.figure()
    for i in range(0, 15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(imgs[i + 20], cmap='Greys')
        plt.title(labels[i + 20])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def process_data_for_softmax_model():
    (train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    train_imgs = train_imgs.astype(np.float)
    test_imgs = test_imgs.astype(np.float)

    train_imgs /= 255
    test_imgs /= 255

    # 图片reshape成28 * 28 = 784 的一维向量
    train_x = train_imgs.reshape((train_imgs.shape[0], 784))
    test_x = test_imgs.reshape((test_imgs.shape[0], 784))

    # label规范为one_hot类型
    num_classes = 10
    train_y = np_utils.to_categorical(train_labels, num_classes)
    test_y = np_utils.to_categorical(test_labels, num_classes)
    return train_x, train_y, test_x, test_y

def create_softmax_model():
    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation("relu"))

    model.add(Dense(512))
    model.add(Activation("relu"))

    model.add(Dense(10))
    model.add(Activation("softmax"))
    return model


def process_data_for_cnn_model():
    (train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    train_imgs = train_imgs.astype(np.float)
    test_imgs = test_imgs.astype(np.float)

    train_imgs /= 255
    test_imgs /= 255

    shape = (28, 28, 1)
    if kbackend.image_data_format() == "channels_last":
        train_x = train_imgs.reshape((train_imgs.shape[0], 28, 28, 1))
        test_x = test_imgs.reshape((test_imgs.shape[0], 28, 28, 1))
    else:
        train_x = train_imgs.reshape((train_imgs.shape[0], 1, 28, 28))
        test_x = test_imgs.reshape((test_imgs.shape[0], 1, 28, 28))
        shape = (1, 28, 28)

    # label规范为one_hot类型
    num_classes = 10
    train_y = np_utils.to_categorical(train_labels, num_classes)
    test_y = np_utils.to_categorical(test_labels, num_classes)
    return train_x, train_y, test_x, test_y, shape

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model


# train_x, train_y, test_x, test_y = process_data_for_softmax_model()
# # create model with keras
# model = create_softmax_model()

train_x, train_y, test_x, test_y, img_shape = process_data_for_cnn_model()
# create model with keras
model = create_cnn_model(img_shape)

# 指定损失函数，测量指标， 优化器
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(train_x, train_y,
          batch_size=128,
          epochs=5,
          verbose=2,
          validation_data=(test_x, test_y))


print(history.history.keys())
print(history.history.values())

# 使用模型进行类别预测
predict_result = model.predict_classes(test_x)
print(predict_result[1])

# save and load
import os
save_root_path = "./model_archives"
if not os.path.exists(save_root_path):
    os.mkdir(save_root_path)
save_path = os.path.join(save_root_path, 'minist.model')
if os.path.exists(save_path):
    os.system("rm -rf {0}".format(save_path))
model.save(save_path)
print("save model to {0}".format(save_path))

loaded_model = models.load_model(save_path)
loss_and_acc = loaded_model.evaluate(test_x, test_y)
print(loss_and_acc)