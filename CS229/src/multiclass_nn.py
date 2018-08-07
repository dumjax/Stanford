import sys, os
sys.path.insert(0, "/afs/.ir.stanford.edu/users/p/k/pkawthek/deep-learning-models/")

import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from vgg16 import VGG16
from imagenet_utils import preprocess_input, decode_predictions
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

def load_labels():

	reader = csv.reader(open(info_file, "r"))
	next(reader, None)
	all_labels = {}
	for row in reader:
		all_labels[row[11]] = row[0]
	labels = [all_labels[img_file] for img_file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_file))]
	unique_labels = list(set(labels))
	painters = dict((unique_labels[idx], idx) for idx in range(len(unique_labels)))

	y = np.array([painters[label] for label in labels])
	y = to_categorical(y)

	return y, painters

def load_features(dims=(224, 224)):

	print("Featurizing.")

	featurizer = VGG16(weights='imagenet', include_top=False)

	for img_file in os.listdir(img_dir):

		img = image.load_img(os.path.join(img_dir, img_file), target_size=dims)

		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		x = featurizer.predict(img)

		try:
  			X
		except NameError:
			X = [x[0]]
		else:
			X.append(x[0])

		print(float(len(X))/total_images, end="\r")

	return np.array(X)

def load_dataset(test_ratio=0.2, rebuild=False, save=True):

	if not rebuild:

		X = np.load(open(dataset_dir + 'X.npy', 'rb'))
		y = np.load(open(dataset_dir + 'y.npy', 'rb'))

	else:

		X = load_features()
		y, painter_idxs = load_labels()

		if save:
			np.save(open(dataset_dir + 'X.npy', 'wb'), X)
			np.save(open(dataset_dir + 'y.npy', 'wb'), y)
			np.save(open(dataset_dir + 'painters.npy', 'wb'), y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rseed)
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_ratio, random_state=rseed)

	return X_train, X_validation, X_test, y_train, y_validation, y_test

def nn(input_shape, output_shape):

	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(output_shape, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

## initialization

rseed = 42

info_file = '../data/all_data_info.csv'

data_dir = '../data/train_top10/'
img_dir = data_dir + 'images/'
dataset_dir = data_dir + 'dataset/'
weights_dir = data_dir + 'weights/'
weights_path = weights_dir + 'weights.h5'
checkpoint = True

total_images = 	len([img_file for img_file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_file))])
print("Total images: ", total_images)

## learning

X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset(test_ratio=0.2, rebuild=False)

# print(X_train.shape, X_valid.shape, X_test.shape)
# print(y_train.shape, y_valid.shape, y_test.shape)

input_shape = X_train.shape[1:]
output_shape = y_train.shape[1]

model = nn(input_shape, output_shape)

if checkpoint:
	checkpoint = ModelCheckpoint(filepath=weights_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
else:
	checkpoint = None

model.fit(X_train, y_train, nb_epoch=100, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[checkpoint])

if not checkpoint:
	model.save_weights(weights_path)

y_pred_test = model.predict(x_test)
np.save(open(dataset_dir + 'y_pred_test.npy', "wb"), y_pred_test)

# categorical_accuracy(y_test, y_pred_test)