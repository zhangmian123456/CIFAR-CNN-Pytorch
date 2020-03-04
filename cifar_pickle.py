"""0"""
import pickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CHANNEL = 3
WIDTH = 32
HEIGHT = 32

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']


def pickle_read():
    imgs_data = []
    imgs_labels = []
    imgs_name = []
    for i in range(5):
        with open("cifar-10-python/data_batch_" + str(i + 1), mode='rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
            imgs_data += list(data_dict[b'data'])
            imgs_labels += list(data_dict[b'labels'])
            imgs_name += list(data_dict[b'filenames'])
    return imgs_data, imgs_labels, imgs_name


def img_read(img_id):
    imgs_data, imgs_labels, imgs_name = pickle_read()
    imgs = np.reshape(imgs_data, [-1, CHANNEL, WIDTH, HEIGHT])

    img_data = imgs[img_id]
    img_label = imgs_labels[img_id]
    img_name = imgs_name[img_id]

    return classification[img_label], img_data, img_name


def rgb_draw(img_id):
    img_label, img_data, img_name = img_read(img_id)
    r = img_data[0]
    g = img_data[1]
    b = img_data[2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)

    img_rgb = Image.merge("RGB", (ir, ig, ib))

    print(img_name)
    plt.imshow(img_rgb)
    plt.show()


def img_save(save_path):
    imgs_data, imgs_labels, imgs_name = pickle_read()
    channel = 3
    width = 32
    height = 32

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img = np.reshape(imgs_data, [-1, channel, width, height])

    for i in range(img.shape[0]):
        r = img[i][0]
        g = img[i][1]
        b = img[i][2]

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)

        rgb = Image.merge("RGB", (ir, ig, ib))

        img_name = "img-" + str(i) + "-" + classification[imgs_labels[i]] + ".png"
        rgb.save(save_path + img_name, "PNG")
