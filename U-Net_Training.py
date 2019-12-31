# Author:Artisan.Fu #email:17862003017@163.com


from __future__ import print_function
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge,Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.models import load_model
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Reshape, Activation, core, \
    Permute
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import os
import configparser as ConfigParser
import warnings  
from base_functions import get_train_data
import h5py
from keras.models import model_from_json
from keras.optimizers import SGD
#from IPython.terminal.tests.test_help import test_profile_list_help

warnings.filterwarnings("ignore")
#K.set_image_dim_ordering('th')
K.image_data_format() == 'channels_last'
# --read configuration file-- #
config = ConfigParser.RawConfigParser()
config.read('./configuration.txt')

# --get parameters-- #
path_local = config.get('unet_parameters', 'path_local')
train_images_dir = path_local + config.get('unet_parameters', 'train_images_dir')
train_labels_dir = path_local + config.get('unet_parameters', 'train_labels_dir')
img_h = int(config.get('unet_parameters', 'img_h'))
img_w = int(config.get('unet_parameters', 'img_w'))
N_channels = int(config.get('unet_parameters', 'N_channels'))
C = int(config.get('unet_parameters', 'C'))

if C > 2:
    gt_list = eval(config.get('unet_parameters', 'gt_gray_value_list'))
else:
    gt_list = None


# --Build a net work-- #
def get_net():
    inputs = Input(shape=(img_h, img_w, N_channels))
	# 网络结构定义
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)


    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(C, (1, 1), activation='relu', kernel_initializer='he_normal')(conv9)

    reshape = Reshape((C, img_h * img_w), input_shape=(C, img_h, img_w))(conv10)
    reshape = Permute((2, 1))(reshape)

    activation = Activation('softmax')(reshape)

    model = Model(input=inputs, output=activation)

    model.compile(optimizer=Adam(lr=1.0e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
print('-' * 30)
print('Loading and pre-processing train data...')
print('-' * 30)

train_x, train_y = get_train_data(train_images_dir, train_labels_dir, img_h, img_w, C=C, gt_list=gt_list)

print('train_x size: ', np.shape(train_x))
print('train_y size: ', np.shape(train_y))

assert(train_y.shape[1] == img_h and train_y.shape[2] == img_w)
train_y = np.reshape(train_y, (train_y.shape[0], img_h*img_w, C))
model_path = path_local+config.get('unet_parameters', 'unet_model_dir')

# --Check whether the output path of the model exists or not-- #
if os.path.isdir(model_path):
    pass
else:
    os.mkdir(model_path)
# ------------------------------------ #
print('-' * 30)
print('Creating and compiling model...')
print('-' * 30)
model = get_net()
model_checkpoint = ModelCheckpoint(model_path + '/unet.hdf5', monitor='loss', save_best_only=True)

print('-' * 30)
print('Fitting model...')
print('-' * 30)
batch_size = int(config.get('unet_parameters', 'batch_size'))
epochs = int(config.get('unet_parameters', 'N_epochs'))
val_rate = config.get('unet_parameters', 'validation_rate')
hist = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True,
                 validation_split=float(val_rate), callbacks=[model_checkpoint], initial_epoch=0)
with open(model_path + '/unet.txt', 'w') as f:
    f.write(str(hist.history))
print('-' * 30)
print('Loading saved weights...')
print('-' * 30)
model.load_weights(model_path + '/unet.hdf5')
json_string = model.to_json()  # equal to: json_string = model.get_config()
open(model_path + '/unet.json', 'w').write(json_string)
model.save_weights(model_path + '/unet_weights.h5')
