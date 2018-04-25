import math
import sys
import keras
import numpy as np
import os
import h5py
#import theano
#import theano.sandbox.cuda.dnn
import scipy.misc
from mean_iou_evaluate import read_masks, mean_iou_score
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.models import Model, load_model
from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf

n_epochs = 1
train_size = 10 #2313
valid_size = 257
load = 0

def read_image(data_size, folder):
    x = np.zeros((data_size, 512, 512, 3))
    y = np.zeros((data_size, 512, 512, 7))
    
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
    x = Conv2DTranspose(7 , kernel_size=(64,64) ,  strides=(32,32) , 
                        padding='same', use_bias=False ,  data_format='channels_last')(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model

def training(model, X_train_, Y_train):
    model.compile(loss=binary_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X_train, Y_train, 
          batch_size=1, epochs=n_epochs, verbose=1)
    model.save('my_model.h5')
    del model 

def validation(model, X_valid, Y_valid):
    #score = model.evaluate(X_test, Y_test, verbose=0)
    output = model.predict(X_valid, batch_size=None, verbose=0, steps=None)
    for n in range(output.shape[0]):
        output_image = np.zeros((512, 512, 3))
        label = np.argmax(output[n], axis=2)
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

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def evaluation(ground_truth_folder, predict_folder):
    pred = read_masks(ground_truth_folder)
    labels = read_masks(predict_folder)
    score = mean_iou_score(pred, labels)
    print("mean iou score: " , score)

if __name__ == '__main__':
    train_folder = './hw3-train-validation/train/'
    ground_truth_folder = './hw3-train-validation/validation/'
    predict_folder = './output/'
    (X_train, Y_train) = read_image(train_size, train_folder)
    (X_valid, Y_valid) = read_image(valid_size, ground_truth_folder)
    if load == 1:
        model = load_model('my_model.h5')
    else:
        model = construct_model()
        training(model, X_train, Y_train)
        model = load_model('my_model.h5')
    validation(model, X_valid, Y_valid)
    evaluation(ground_truth_folder, predict_folder)
