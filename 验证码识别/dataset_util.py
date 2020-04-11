import os
import random
from captcha import image
import numpy as np
import PIL.Image as PILImage
import cv2
import shutil

def create_train_dataset(output_dir: str, width: float, height: float, captcha_count=4, count=2000):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    number_sets = lambda offset: str(offset)
    lowcase_char_sets = lambda offset: chr(97 + offset)
    upcase_char_sets = lambda offset: chr(65 + offset)

    avaliable_sets = []
    for i in range(0, 10):
        avaliable_sets.append(number_sets(i))
    # for i in range(0, 26):
    #     avaliable_sets.append(lowcase_char_sets(i))
    # for i in range(0, 26):
    #     avaliable_sets.append(upcase_char_sets(i))

    def random_str(count):
        str = ""
        for i in range(0, count):
            rand_index = random.randrange(0, len(avaliable_sets) - 1)
            str = str + avaliable_sets[rand_index]
        return str

    image_captcha = image.ImageCaptcha(width=width, height=height)
    for i in range(count):
        captcha_str = random_str(captcha_count)
        image_captcha.write(captcha_str, output_dir + "/" + captcha_str + ".png", "png")
        print("Gen captcha: {0}".format(captcha_str))

def remove_dataset(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def read_dataset(dir):
    images = []
    labels = []
    for subpath in os.listdir(dir):
        if subpath.endswith(".png"):
            image = np.array(PILImage.open(os.path.join(dir, subpath)))
            label = subpath[:-4]
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images.append(gray_img / 255.0)
            labels.append(label)
    return images, labels


