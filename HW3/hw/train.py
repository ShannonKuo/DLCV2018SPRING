import math
import sys
import pickle
import keras
import numpy as np
import os
import theano
import theano.sandbox.cuda.dnn
import cv2

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model

nb_epoch_1 = 300
batch_size = 32
valid = 50
semi = 1
loadModel = 0

train_size = 2312
valid_size = 256

def read_image(data_size, folder):
    x = np.zeros((data_size, 3, 512, 512))
    y = np.zeros((data_size, 7, 512, 512))
    
    for i in range(data_size):
        index = str(i).zfill(4)
        img = cv2.imread(folder + index + '_sat.jpg')
        x(i) = img
    x = np.reshape(x, (data_size, 3, 512, 512))

    for i in range(data_size):
        index = str(i).zfill(4)
        img = cv2.imread(argv[1] + index + '_mask.png')
        img = (img >= 128).astype(int)
        img = 4 * img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2]
        y[i, j, img == j]

        y[i, 0, img == 3] = 1  # (Cyan: 011) Urban land 
        y[i, 1, img == 6] = 1  # (Yellow: 110) Agriculture land 
        y[i, 2, img == 5] = 2  # (Purple: 101) Rangeland 
        y[i, 3, img == 2] = 3  # (Green: 010) Forest land 
        y[i, 4, img == 1] = 4  # (Blue: 001) Water 
        y[i, 5, img == 7] = 5  # (White: 111) Barren land 
        y[i, 6, img == 0] = 1  # (Black: 000) Unknown 

    return (x, y)

def construct_model()
    img_input = Input(shape=(512, 512, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(img_input, x)
    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model

if __name__ == '__main__':
    (x_train, y_train) = read_image(train_size, './hw3-train-validation/train')
    (x_valid, y_valid) = read_image(valid_size, './hw3-train-validation/validation')
    model = construct_model()
    training(model, x_train, y_train)
