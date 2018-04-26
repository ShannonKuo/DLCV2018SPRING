import math
import sys
import keras
import numpy as np
import os
import h5py
import scipy.misc
from mean_iou_evaluate import read_masks, mean_iou_score
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.models import Model, load_model
from keras.objectives import *
from keras.metrics import binary_crossentropy
from keras.callbacks import Callback

import keras.backend as K
import tensorflow as tf

n_epochs = 40
train_size = 2313
valid_size = 257
load = 1
model_name = 'my_model_epoch15_cat.h5'
save_model_name = 'my_model_epoch55_cat.h5'

def read_image(data_size, folder):
    x = np.zeros((data_size, 512, 512, 3))
    y = np.zeros((data_size, 512, 512, 7), dtype=bool)
    
    for i in range(data_size):
        index = str(i).zfill(4)
        file_path = folder + index + '_sat.jpg'
        img = scipy.misc.imread(file_path)
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

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='block6_conv2')(x)
    x = Conv2D(7, (1, 1), activation='relu', padding='same', name='block6_conv3')(x)
    x = Conv2DTranspose(7 , kernel_size=(64,64) ,  strides=(32,32) , 
                        padding='same', use_bias=False ,  data_format='channels_last')(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model

def training(model, X_train_, Y_train, X_valid):
    metrics = Metrics(X_valid)
    keras.optimizers.Adadelta(lr = 1.0, rho = 0.95, epsilon = 1e-06)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X_train, Y_train, 
          batch_size=4, epochs=n_epochs, verbose=1, callbacks=[metrics])
    return model

def validation(model, X_valid):
    print("validation")
    output = model.predict(X_valid, batch_size=1, verbose=0, steps=None)
    print(output.shape)
    #labels = np.zeros((output.shape[0], 512, 512))
    for n in range(output.shape[0]):
        output_image = np.zeros((512, 512, 3))
        label = np.argmax(output[n], axis=2)
        #labels[n] = label
        for i in range(512):
            for j in range(512):
                ans = label[i, j]
                if ans == 0:
                    output_image[i, j] = [0, 255, 255]
                elif ans == 1:
                    output_image[i, j] = [255, 255, 0]
                elif ans == 2:
                    output_image[i, j] = [255, 0, 255]
                elif ans == 3:
                    output_image[i, j] = [255, 0, 255]
                elif ans == 4:
                    output_image[i, j] = [0, 0, 255]
                elif ans == 5:
                    output_image[i, j] = [255, 255, 255]
                elif ans == 6:
                    output_image[i, j] = [0, 0, 0]
        index = str(n).zfill(4)
        folder = './output'
        file_path = folder + '/' + index + '_mask.png'
        if not os.path.exists(folder):
            os.makedirs(folder)
        scipy.misc.imsave(file_path, output_image)

class Metrics(Callback):
    def __init__(self, x):
        self.x = x
    def on_epoch_end(self, epoch, logs={}):
        validation(model, self.x)
        pred = read_masks(ground_truth_folder)
        labels = read_masks(predict_folder)
        score = mean_iou_score(pred, labels)
        #pred = validation(model, self.x, self.y)
        #score = mean_iou_score(pred, self.labels)

def evaluation(ground_truth_folder, predict_folder):
    pred = read_masks(ground_truth_folder)
    labels = read_masks(predict_folder)
    score = mean_iou_score(pred, labels)
    return

def evaluation_with_label(pred, labels):
    score = mean_iou_score(pred, labels)
    return
if __name__ == '__main__':
    train_folder = './hw3-train-validation/train/'
    ground_truth_folder = './hw3-train-validation/validation/'
    predict_folder = './output/'
    (X_train, Y_train) = read_image(train_size, train_folder)
    (X_valid, Y_valid) = read_image(valid_size, ground_truth_folder)
    if load == 1:
        model = load_model(model_name)
        model = training(model, X_train, Y_train, X_valid)
        model.save(save_model_name)
        #validation(model, X_valid)
        #evaluation(ground_truth_folder, predict_folder)
    else:
        model = construct_model()
        model = training(model, X_train, Y_train, X_valid)
        model.save(model_name)
        #model = load_model(model_name)
        #labels = validation(model, X_valid, Y_valid)
        #evaluation(ground_truth_folder, predict_folder)
        #evaluation_with_label(Y_train, labels)
