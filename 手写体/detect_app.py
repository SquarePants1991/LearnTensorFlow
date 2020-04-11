from keras import models
import numpy as np
import keras.backend as kbackend
import tools.drawboard as Drawboard
import cv2
import matplotlib.pyplot as plt
import keras.utils as kutil

def process_data_for_cnn_model(img):
    img = img.astype(np.float)
    img /= 255

    shape = (28, 28, 1)
    if kbackend.image_data_format() == "channels_last":
        img = img.reshape(28, 28, 1)
    else:
        img = img.reshape(1, 28, 28)
        shape = (1, 28, 28)
    return img, shape


if __name__ == '__main__':
    loaded_model = models.load_model("./model_archives/minist.model")
    def user_draw_cb(img):
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_img = cv2.resize(new_img, (28, 28))
        new_img, shape = process_data_for_cnn_model(new_img)
        results = loaded_model.predict(np.array([new_img]))
        result = results[0]
        cv2.imshow("gray", new_img)

        print(result.argmax(axis=0))
        print(result)
    draw_board = Drawboard.DrawBoard(280, 280)
    draw_board.run(user_draw_cb)

