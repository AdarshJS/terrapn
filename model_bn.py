# env
# conda install cudatoolkit=10.0.130
# conda install cudnn=7.6.0=cuda10.0_0
# pip install --upgrade tensorflow-gpu

# conda create --name tf22 tensorflow-gpu=2.2 cudatoolkit=10.1 cudnn=7.6 python=3.8 pip=20.0
# conda install -c conda-forge keras matplotlib
# conda install -c anaconda scikit-learn
# pip install Pillow xgboost seaborn tqdm 



from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, add, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
from keras import backend as K
from keras.layers.merge import concatenate
from PIL import Image

# from sklearn.datasets import load_iris
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import Input, Model
import tensorflow as tf

from sklearn.datasets import load_iris
from keras import Input, Model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
	# add max pooling layer
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

def naive_inception_module(layer_in, f1, f2, f3):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
	# 5x5 conv
	conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out




def get_model(v_len, patch_len):

	inputs_i = Input(shape=(patch_len, patch_len, 3,), name='image')
	inputs_v = Input(shape=(v_len * 2,), name='velocity')



	v = Dense(128, kernel_initializer='normal', activation='relu', name='v_1')(inputs_v)
	# v = BatchNormalization()(v)
	v = Dropout(0.5)(v)
	v = Dense(64, kernel_initializer='normal', activation='relu', name='v_2')(v)
	v = BatchNormalization()(v)




	img = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv_1')(inputs_i)

	img = residual_module(img, 64)
	img = residual_module(img, 64)
	img = BatchNormalization()(img)
	img = MaxPooling2D((2,2), strides=(2,2))(img)

	# img_shape = K.int_shape(img)
	# print(img_shape)
	# from IPython import embed;embed()

	img = residual_module(img, 128)
	img = residual_module(img, 128)
	img = BatchNormalization()(img)
	# img = MaxPooling2D((2,2), strides=(2,2))(img)

	# img = residual_module(img, 256)
	# img = residual_module(img, 256)
	# img = BatchNormalization()(img)


	img_shape = K.int_shape(img)
	pool = AveragePooling2D(pool_size=(img_shape[1], img_shape[2]),
								strides=(1, 1))(img)
	flatten = Flatten()(pool)
	img = Dense(units=128, kernel_initializer="he_normal",
					activation="softmax")(flatten)


	# img_shape = K.int_shape(img)
	# print(img_shape)
	# from IPython import embed;embed()
	layer_out = concatenate([img, v], axis=-1)
	layer_out = Dense(500, activation='softmax')(layer_out)
	layer_out = BatchNormalization()(layer_out)

	output = Dense(4, activation='linear', name='out')(layer_out)

	model = Model(inputs=[inputs_i, inputs_v], outputs=output)

	model.compile(loss='mean_absolute_error',
				optimizer='adam',
				metrics=['mean_absolute_error'])

	model.summary()

	return model


v_len = 50
patch_len = 100
model = get_model(v_len, patch_len)
# v = np.load("./dataset/asphalt/input_velocity_1600.npy")[-1 * v_len:]
# v = v.reshape(-1)
# # from IPython import embed;embed()
# i = Image.open("./dataset/asphalt/1600_patch100.png")
# i = np.asarray(i)
# print(model.predict([np.array([i, i, i]), np.array([v, v, v])]))

