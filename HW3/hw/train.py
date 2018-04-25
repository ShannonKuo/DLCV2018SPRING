import math
import sys
import keras
import numpy as np
import os
import h5py
#import theano
#import theano.sandbox.cuda.dnn
#import cv2
import scipy.misc

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model

train_size = 10#2312
valid_size = 10#256

def read_image(data_size, folder):
    #x = np.zeros((data_size, 3, 512, 512))
    #y = np.zeros((data_size, 7, 512, 512))
    x = np.zeros((data_size, 512, 512, 3))
    y = np.zeros((data_size, 512, 512, 7))
    
    for i in range(data_size):
        index = str(i).zfill(4)
        file_path = folder + index + '_sat.jpg'
        img = scipy.misc.imread(file_path)
        #img = np.reshape(img, (3, 512, 512))
        x[i] = img

    x = x.astype('float32')
    x /= 255

    for i in range(data_size):
        index = str(i).zfill(4)
        file_path = folder + index + '_mask.png'
        img = scipy.misc.imread(file_path)
        img = (img >= 128).astype(int)
        img = 4 * img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2]

        y[i, img == 3, 0] = 1  # (Cyan: 011) Urban land 
        y[i, img == 6, 1] = 1  # (Yellow: 110) Agriculture land 
        y[i, img == 5, 2] = 1  # (Purple: 101) Rangeland 
        y[i, img == 2, 3] = 1  # (Green: 010) Forest land 
        y[i, img == 1, 4] = 1  # (Blue: 001) Water 
        y[i, img == 7, 5] = 1  # (White: 111) Barren land 
        y[i, img == 0, 6] = 1  # (Black: 000) Unknown 
    return (x, y)

def construct_model():
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

    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='block6_conv2')(x)
    x = Conv2D(7, (1, 1), activation='relu', padding='same', name='block6_conv3')(x)
    x = Conv2DTranspose(7 , kernel_size=(32,32) ,  strides=(32,32) , use_bias=False ,  data_format='channels_last')(x)

    model = Model(img_input, x)
    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model

def training(model, X_train_, Y_train):
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)

def validation(model, X_valid, Y_valid):
    score = model.evaluate(X_test, Y_test, verbose=0)

if __name__ == '__main__':
    (X_train, Y_train) = read_image(train_size, './hw3-train-validation/train/')
    (X_valid, Y_valid) = read_image(valid_size, './hw3-train-validation/validation/')
    model = construct_model()
    training(model, X_train, Y_train)
    validation(model, X_valid, Y_valid)
