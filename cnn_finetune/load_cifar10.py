import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as kbe
from keras.utils import np_utils


NUM_CLASSES = 10


def load_cifar10_data(img_rows, img_cols, nb_train_samples, nb_valid_samples):
    # Load cifar10 training and validation sets
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()

    # Resize trainging images
    if kbe.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img.transpose(1, 2, 0), (img_rows, img_cols)).transpose(2, 0, 1) for img in x_train[:nb_train_samples, :, :, :]])
        x_valid = np.array([cv2.resize(img.transpose(1, 2, 0), (img_rows, img_cols)).transpose(2, 0, 1) for img in x_valid[:nb_valid_samples, :, :, :]])
    else:
        x_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_train[:nb_train_samples, :, :, :]])
        x_valid = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_valid[:nb_valid_samples, :, :, :]])

    # Transform targets to keras compatible format
    y_train = np_utils.to_categorical(y_train[:nb_train_samples], NUM_CLASSES)
    y_valid = np_utils.to_categorical(y_valid[:nb_valid_samples], NUM_CLASSES)

    return x_train, y_train, x_valid, y_valid
