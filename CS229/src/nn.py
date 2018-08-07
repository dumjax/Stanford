import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import csv

data_dir = '../data/painters_2/'
feature_dir = data_dir + 'features/'

# path to the model weights file.
weights_path = 'pretrained_models/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'

# dimensions of our images.
img_width, img_height = 224, 224

train_count_per_painter = 300
valid_count_per_painter = 76

nb_train_samples = 600
nb_validation_samples = 152
nb_epoch = 20

def train_top_model():

	x_1 = np.load(feature_dir + 'features_Rembrandt.npy')
	x_2 = np.load(feature_dir + 'features_Pablo Picasso.npy')

	train_data = np.concatenate((x_1[:300, :, :, :], x_2[:300, :, :, :]), axis=0)
	validation_data = np.concatenate((x_1[train_count_per_painter:train_count_per_painter + valid_count_per_painter, :, :, :], x_2[train_count_per_painter:train_count_per_painter + valid_count_per_painter, :, :, :]), axis=0)

	train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
	validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(train_data, train_labels, nb_epoch=nb_epoch, batch_size=32, validation_data=(validation_data, validation_labels))
	
	model.save_weights(top_model_weights_path)

train_top_model()