import keras.backend as kb
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Concatenate
import keras

# 将图片数据规范化为keras可以处理的格式
def normalize_images(images):
    np_images = np.array(images)
    shape = (np_images.shape[1], np_images.shape[2], 1)
    if kb.image_data_format() == 'channels_last':
        np_images = np_images.reshape((np_images.shape[0], np_images.shape[1], np_images.shape[2], 1))
    else:
        shape = (1, np_images.shape[1], np_images.shape[2])
        np_images = np_images.reshape((np_images.shape[0], 1, np_images.shape[1], np_images.shape[2]))
    return np_images, shape


def normalize_labels(labels):
    label_np_array = []
    for label in labels:
        onhots_arrays = []
        for i in range(0, len(label)):
            ch = label[i]
            num = int(ch)
            onhots_arrays.append(np_utils.to_categorical(num, 10))
        onehot_label = np.concatenate(onhots_arrays)
        label_np_array.append(onehot_label)
    return np.array(label_np_array)


def label_string_from_onehot(onhot_array):
    label = ""
    for i in range(0, 4):
        onehot_seg = onhot_array[i * 10: (i + 1) * 10]
        index = 0
        for j in range(0, len(onehot_seg)):
            if onehot_seg[j] == 1:
                label += str(index)
                break
            index += 1
    return label


def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(conv1)
    x = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(conv3)
    # x = Dense(128, activation="softmax")(x)
    x = Flatten()(x)
    x = Dropout(rate=0.25)(x)
    category_denses = [Dense(10, activation="softmax")(x), Dense(10, activation="softmax")(x), Dense(10, activation="softmax")(x), Dense(10, activation="softmax")(x)]
    outputs = Concatenate()(category_denses)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


import keras.utils.vis_utils as plot_util
import pathlib
import os.path
def save_model_flow_as_image(model, save_path):
    parent_dir = pathlib.Path(save_path).parent.absolute()
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    plot_util.plot_model(model, save_path)


import matplotlib.pyplot as plt
import pickle
def save_history(history, save_path):
    parent_dir = pathlib.Path(save_path).parent.absolute()
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    with open(save_path, "wb") as file:
        pickle.dump(history, file)

def load_history(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def plot_train_history(history):
    # history.history["loss"]
    # history.history["accuracy"]
    # history.history["val_loss"]
    # history.history["val_accuracy"]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history["loss"], color="red", label="train loss")
    plt.plot(history.history["val_loss"], color="green", label="test loss")
    plt.title("loss")
    plt.subplot(2, 1, 2)
    plt.plot(history.history["accuracy"], color="red", label="train accuracy")
    plt.plot(history.history["val_accuracy"], color="green", label="test accuracy")
    plt.title("accuracy")
    plt.show()
