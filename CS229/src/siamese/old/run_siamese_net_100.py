
# coding: utf-8

# Data organizer stack

# In[1]:
from __future__ import absolute_import

import os
from os import mkdir, listdir, makedirs
from os.path import join, abspath, basename, splitext, dirname, isdir
import numpy as np
np.random.seed(42)
from json import load, dump
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D 
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
import threading
import random
from random import shuffle
from itertools import combinations, chain, product, izip
from math import ceil
from utils import *
import pickle
from multiprocessing import Pool
import time
from sklearn.metrics import *
from keras.utils.visualize_util import plot

NUM_CLASSES = 100
IMGS_DIM_3D = (3, 256, 256)
IMGS_DIM_2D = IMGS_DIM_3D[1:]

data_dir = '/data/paintersbynumbers/'

models_dir = join(data_dir, 'models/siamese')
dir_tr = join(data_dir, 'train_{:d}_100'.format(IMGS_DIM_2D[0]))
dir_val = join(data_dir, 'val_{:d}_100'.format(IMGS_DIM_2D[0]))

def generators(dir_tr, dir_val, batch_size, num_samples_per_cls=1, num_samples_per_cls_val=None):

	# gen_tr = PairsImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	
	# gen_val = PairsImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

	gen_tr = PairsImageDataGenerator()
	gen_val = PairsImageDataGenerator()

	# gen_tr = PairsImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	# gen_val = PairsImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

	sample = np.array(apply_to_images_in_subdirs(dir_tr, load_img_arr, num_samples_per_cls=num_samples_per_cls))
	gen_tr.fit(sample)
	gen_val.fit(sample)

	gen_tr = gen_tr.flow_from_directory(dir_tr, batch_size=batch_size)
	gen_val = gen_val.flow_from_directory(dir_val, batch_size=batch_size, num_samples_per_cls=num_samples_per_cls_val, is_val=True)

	return gen_tr, gen_val
	
class PairsImageDataGenerator(ImageDataGenerator):

	def __init__(self, *args, **kwargs):
		super(PairsImageDataGenerator, self).__init__(*args, **kwargs)

	def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
			 save_to_dir=None, save_prefix='', save_format='jpg', num_samples_per_cls=None):
		raise NotImplementedError

	def flow_from_directory(
			self, dir_, target_size=IMGS_DIM_2D, color_mode='rgb',
			classes=None, class_mode='categorical', batch_size=32,
			shuffle=True, seed=None, save_to_dir=None, save_prefix='',
			save_format='jpg', num_samples_per_cls=None, is_val=False):

		return PairsDirectoryIterator(
			dir_, self, batch_size, num_samples_per_cls, is_val)


def _std_random_transform_img(img, imgen):
	img = imgen.random_transform(img)
	return imgen.standardize(img)

def imgreader(fns, imgen):
	if imgen:
		return _std_random_transform_img(load_img_arr(fns[0]), imgen), _std_random_transform_img(load_img_arr(fns[1]), imgen)
	else:
		return load_img_arr(fns[0]), load_img_arr(fns[1])

def imgreader_pool_wrapper(args):
	return imgreader(*args)

class PairsDirectoryIterator(object):

	def __init__(self, dir_, image_data_generator,
				 batch_size=32, num_samples_per_cls=None, is_val=False):
	
		self.dir_ = dir_
		self.paths, self.y = self._get_paths_labels_from_dir(dir_, num_samples_per_cls)
		self.batch_size = batch_size
		self._init_pairs_generator()
		self.image_data_generator = image_data_generator
		self.lock = threading.Lock()
		self.pool = Pool(processes=8)
		self.is_val = is_val

	@staticmethod
	def _get_paths_labels_from_dir(dir_, num_per_cls):
		def path_label(p): return [p, basename(dirname(p))]
		paths_labels = apply_to_images_in_subdirs(dir_, path_label, num_samples_per_cls=num_per_cls)
		paths_labels = np.array(paths_labels)
		return paths_labels[:, 0], paths_labels[:, 1].astype(int)

	def _init_pairs_generator(self):
		self.pairs_generator = alt_pairs_generator(self.dir_, self.batch_size, lambda a, b: [a, b])

	def iter(self):
		return self

	def next(self):

		start = time.time()

		with self.lock:
			try:
				paths_batch, y_batch = next(self.pairs_generator)
			except StopIteration:
				self._init_pairs_generator()
				paths_batch, y_batch = next(self.pairs_generator)
		X_batch = self.pool.map(imgreader_pool_wrapper, [(paths_pair, None) for paths_pair in paths_batch])
		X_batch = np.array(X_batch)/255.0
		X_batch = [X_batch[:, 0], X_batch[:, 1]]

		if self.is_val:
			return y_batch, paths_batch
		else:
			# print "Processed batch: {0}".format(time.time()-start)
			return X_batch, y_batch

def alt_pairs_generator(dir_, batch_size, pair_func):
		
	classes = range(NUM_CLASSES)
	
	negative_pairs = combinations(classes, 2)
	positive_pairs = izip(classes, classes)
	
	while True:
		X_batch, y_batch = [], []
		
		for i in range(batch_size/2):
			
			pairs = []
			try:
				pairs.append(next(positive_pairs))
				pairs.append(next(negative_pairs))
			except StopIteration:
				return
			
			for pair in pairs:
				class_a, class_b = int(pair[0]), int(pair[1])     
				
				subdir_a = join(dir_, str(class_a))
				subdir_b = join(dir_, str(class_b))
				
				images_a = [os.path.join(subdir_a, f) for f in os.listdir(subdir_a) 
							if os.path.isfile(os.path.join(subdir_a, f))]
				images_b = [os.path.join(subdir_b, f) for f in os.listdir(subdir_b) 
							if os.path.isfile(os.path.join(subdir_b, f))]
				
				if len(images_a) > 0 and len(images_b) > 0:
					x_a = random.choice(images_a)
					x_b = random.choice(images_b)              
				
					X_batch.append(pair_func(x_a, x_b))
					y_batch.append(float(class_a != class_b))
					
		X_batch = np.array(X_batch)
		y_batch = np.array(y_batch)
					
		yield X_batch, y_batch

def apply_to_images_in_subdirs(parent_dir, func, num_samples_per_cls=None, **kwargs):
	results = []
	for cls_dir_name in listdir(parent_dir):
		cls_dir = abspath(join(parent_dir, cls_dir_name))
		r = _apply_to_first_n_in_dir(func, cls_dir, num_samples_per_cls, **kwargs)
		results += r
	return results

def _apply_to_first_n_in_dir(func, dir_, num_samples_per_cls, **kwargs):
	if not isdir(dir_):
		return []
	results = []
	for path in listdir(dir_)[:num_samples_per_cls]:
		abspath_ = abspath(join(dir_, path))
		result = func(abspath_, **kwargs)
		results.append(result)
	return results

BATCH_SIZE = 16

MAX_EPOCHS = 50
TRAIN_SAMPLES_PER_EPOCH = 128
TOTAL_TRAIN_SAMPLES = TRAIN_SAMPLES_PER_EPOCH * MAX_EPOCHS

NUM_SAMPLES_PER_CLS = 1

MARGIN = 1

def build_siamese_net():
	
	processor = _shared_net(full=False)

	input_1 = Input(shape=IMGS_DIM_3D)
	input_2 = Input(shape=IMGS_DIM_3D)

	processed_1 = processor(input_1)
	processed_2 = processor(input_2)

	distance = Lambda(_euclidean_distance, output_shape=_eucl_dist_output_shape)([processed_1, processed_2])

	model = Model(input=[input_1, input_2], output=distance)
	
	model.compile(loss=_contrastive_loss, optimizer='rmsprop')

	return model

def train_model():
			
	hist = model.fit_generator(
		generator=gen_tr,
		nb_epoch=MAX_EPOCHS,
		samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
		verbose=2
	)
	
	return hist
	
def _shared_net(full=False):
		
	nb_filters = 32
	nb_pool = 2
	nb_conv = 3
	
	if full:
	
		seq = Sequential()
		seq.add(Dense(128, input_shape=IMGS_DIM_3D, activation='relu'))
		seq.add(Dropout(0.1))
		seq.add(Dense(128, activation='relu'))
		seq.add(Dropout(0.1))
		seq.add(Dense(128, activation='relu'))
		model = seq
		
	else:
		model = Sequential()
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv,  activation='relu',
								input_shape=IMGS_DIM_3D, border_mode='valid'))
		# model.add(BatchNormalization(axis=1, mode=2))
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
		# model.add(BatchNormalization(axis=1, mode=2))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		model.add(Dropout(p=0.1))
		
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(p=0.1))
		# model.add(BatchNormalization(mode=2))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(p=0.1))
		# model.add(BatchNormalization(mode=2))     
	
	plot(model, to_file='shared.png')

	return model

def _euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def _eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return shape1
				  
def _contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = MARGIN
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(y_pred, y_true):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return y_true[y_pred.ravel() < 0.5].mean()

def save_model(model):
	arch_fname = model_fname + '.arch.json'
	weights_fname = model_fname + '.weights.h5'
	open(arch_fname, 'w').write(model.to_json())
	model.save_weights(weights_fname, overwrite=True)

def load_model():
	arch_fname = model_fname + '.arch.json'
	weights_fname = model_fname + '.weights.h5'
	model_json_string = open(arch_fname).read()
	model=model_from_json(model_json_string)
	model.load_weights(weights_fname)
	return model

def gen_test(regen_test=False):

	y_test = []
	path_pairs = []
	while True:
		data = next(gen_val)
		if data:
			y_batch, pairs = data
			y_test += list(y_batch)
			path_pairs += list(pairs)
		else:
			break
		if len(y_test) > TOTAL_VAL_SAMPLES:
			break
	return np.array(y_test), np.array(path_pairs)

def test_model(metrics, regen_test=False):

	if regen_test:
		y_test, path_pairs = gen_test()
		np.save(join(data_dir, 'features/y_test'), y_test)
		np.save(join(data_dir, 'features/test_pairs'), path_pairs)
		print "Generated test"
	else:
		y_test = np.load(join(data_dir, 'features/y_test.npy'))
		path_pairs = np.load(join(data_dir, 'features/test_pairs.npy'))
		print "Loaded test"
	y_test_pred = []
	perf = {}
	pool = Pool(processes=4)
	batch_size = 16
	num_batches = path_pairs.shape[0]/batch_size
	test_batches = num_batches
	for i in range(test_batches): # num_batches
		print "Test batch : {0}".format(i)
		batch = range(i * batch_size, (i+1) * batch_size)
		pairs_batch = path_pairs[batch, :]
		X_batch = pool.map(imgreader_pool_wrapper, [(pair, None) for pair in pairs_batch])
		X_batch = np.array(X_batch)
		X_batch = [X_batch[:, 0], X_batch[:, 1]]
		y_test_pred += list(model.predict(X_batch))
	y_test = y_test[:test_batches*batch_size]
	y_test_pred = [int(y.ravel() > 0.5) for y in y_test_pred]
	for metric in metrics:
		if metric == 'f1':
			val = f1_score(y_test, y_test_pred)
		elif metric == 'precision':
			val = precision_score(y_test, y_test_pred)
		elif metric == 'recall':
			val = recall_score(y_test, y_test_pred)
		elif metric == 'accuracy':
			val = accuracy_score(y_test, y_test_pred)
		elif metric == 'roc_auc':
			val = roc_auc_score(y_test, y_test_pred)
		perf[metric] = val
	return y_test, y_test_pred, perf

# Build and train siamese net

gen_tr, gen_val = generators(dir_tr, dir_val, batch_size=BATCH_SIZE, num_samples_per_cls=NUM_SAMPLES_PER_CLS)
print "\nBuilt generators"

rebuild_model=True
regen_test = False

model_fname = join(models_dir, 'siamese_100_1210')
if rebuild_model:
	model = build_siamese_net()
	print "Built model"
	# hist = train_model()
	# print "Trained model"
	# save_model(model)
else: 
	model = load_model()
	print "Loaded trained model"

plot(model, to_file='siamese.png')

# metrics = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']
# y_test, y_test_pred, perf = test_model(metrics, regen_test=regen_test)
# print y_test
# print y_test_pred
# print perf