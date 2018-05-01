import math
import sys
import keras
import numpy as np
import os
import h5py
import scipy.misc
from mean_iou_evaluate import read_masks, mean_iou_score
from keras.layers import * #Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.objectives import *
from keras.metrics import binary_crossentropy
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
import keras.backend as K
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#baseline 0.642
#python3 train.py 18 0 my_model_epoch30_softmax_lr0.5.h5 my_model_epoch30_lr0.5_softmax.h5 Adadelta output_mean_softmax_lr0.5.txt softmax 0.5

n_epochs = 25#int(sys.argv[1])
train_size =2313
valid_size = 257
load = 0#int(sys.argv[2])
model_name = sys.argv[1]
save_model_name = "model.h5"#sys.argv[4]
optimizer = "Adadelta"#sys.argv[5]
output_file_name = "output.txt"#sys.argv[6]
data_augmentation = 0
activation = "softmax"#sys.argv[7]
learning_rate = 0.5#float(sys.argv[8])
train_test = sys.argv[2]
model_type = sys.argv[3]
IMAGE_ORDERING = 'channels_last' 
ground_truth_folder = sys.argv[4]#'./hw3-train-validation/validation/'
predict_folder = sys.argv[5]#'./output/'

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
     featurewise_center=False,  # set input mean to 0 over the dataset
     samplewise_center=False,  # set each sample mean to 0
     featurewise_std_normalization=False,  # divide inputs by std of the dataset
     samplewise_std_normalization=False,  # divide each input by its std
     zca_whitening=False,  # apply ZCA whitening
     rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
     horizontal_flip=True,  # randomly flip images
     vertical_flip=False) # randomly flip images



def read_image(data_size, folder):
    x = np.zeros((data_size, 512, 512, 3))
    y = np.zeros((data_size, 512, 512, 7), dtype=bool)
    
    file_list = [file for file in os.listdir(folder) if file.endswith('_sat.jpg')]
    file_list.sort()

    for i, file in enumerate(file_list):
        img = scipy.misc.imread(os.path.join(folder, file))
        x[i] = img

    x = x.astype('float32')

    file_list = [file for file in os.listdir(folder) if file.endswith('_mask.png')]
    file_list.sort()

    for i, file in enumerate(file_list):
        img = scipy.misc.imread(os.path.join(folder, file))
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
    x = Activation(activation)(x)

    model = Model(img_input, x)
    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    #print(model.summary())

    return model

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
    
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

    return o1 , o2 

def construct_FCN8():

    #code from
    #https://github.com/divamgupta/image-segmentation-keras/blob/master/Models/FCN8.py
    #https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5

    nClasses = 7
    input_height = 512
    input_width = 512
    vgg_level = 3

    
    img_input = Input(shape=(input_height,input_width,3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING, trainable=False )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING, trainable=False )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    f5 = x


    o = f5

    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING, padding='same' )(o)

    o2 = f4
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , padding='same', data_format=IMAGE_ORDERING))(o2)
    
    o , o2 = crop( o , o2 , img_input )
    
    o = Add()([ o , o2 ])

    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING, padding='same' )(o)
    o2 = f3 
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , padding='same', data_format=IMAGE_ORDERING))(o2)
    o2 , o = crop( o2 , o , img_input )
    o  = Add()([ o2 , o ])


    o = Conv2DTranspose( nClasses , kernel_size=(16,16) ,  strides=(8,8) ,  use_bias=False, data_format=IMAGE_ORDERING, padding='same', )(o)
    
    o_shape = Model(img_input , o ).output_shape
    
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    #print(Model(img_input , o ).summary())
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model
    
    

def training(model, X_train_, Y_train, X_valid):
    metrics = Metrics(X_valid, model)
    keras.optimizers.Adadelta(lr = learning_rate, rho = 0.95, epsilon = 1e-06)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    if (data_augmentation):
        print("data augmentation")
        datagen.fit(X_train)
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                batch_size=2),
                steps_per_epoch=X_train.shape[0],
                epochs=n_epochs,
                callbacks=[metrics])
    else:
        print("without data augmentation")
        model.fit(X_train, Y_train, 
                batch_size=2, epochs=n_epochs, verbose=1, callbacks=[metrics])
    return model

def validation(model, X_valid):
    print("validation")
    output = model.predict(X_valid, batch_size=1, verbose=0, steps=None)

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
                    output_image[i, j] = [0, 255, 0]
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
    def __init__(self, x, model):
        self.x = x
        self.model = model
        self.prev_high_score = 0.0
    def on_epoch_end(self, epoch, logs={}):
        validation(model, self.x)
        pred = read_masks(ground_truth_folder)
        labels = read_masks(predict_folder)
        now_score = mean_iou_score(pred, labels)
        if (now_score >= self.prev_high_score):
            print("save model")
            self.model.save(model_name)
            self.prev_high_score = now_score
        
        file = open(output_file_name, "a+")
        file.write(str(now_score))
        file.write('\n')
        file.close()

def evaluation(model, X_valid):

    validation(model, X_valid)
    pred = read_masks(ground_truth_folder)
    labels = read_masks(predict_folder)
    now_score = mean_iou_score(pred, labels)

    return

if __name__ == '__main__':
    train_folder = './hw3-train-validation/train/'

    if train_test == 'test':
        print("testing...")
        model = load_model(model_name)
        (X_valid, Y_valid) = read_image(valid_size, ground_truth_folder)
        evaluation(model, X_valid)
        
    else:
        (X_train, Y_train) = read_image(train_size, train_folder)
        (X_valid, Y_valid) = read_image(valid_size, ground_truth_folder)
        if load == 1:
            print("loading model...")
            model = load_model(model_name)
            print("training model...")
            model = training(model, X_train, Y_train, X_valid)
        else:
            if model_type == 'best':
                print("constructing best model...")
                model = construct_FCN8()
                
                print("training best model...")
                model = training(model, X_train, Y_train, X_valid)
            else:
                print("constructing baseline model...")
                model = construct_model()
                print("training baseline model...")
                model = training(model, X_train, Y_train, X_valid)
