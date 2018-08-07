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
from itertools import *
import itertools
from math import ceil
from utils import *
import pickle
from multiprocessing import Pool
import time
from sklearn.metrics import *
from keras.utils.visualize_util import plot
import sys
sys.setrecursionlimit(5000)

IMGS_DIM_3D = (3, 64, 64)
IMGS_DIM_2D = IMGS_DIM_3D[1:]

MAX_CLASSES = 100

data_dir = '/data/paintersbynumbers/'

models_dir = join(data_dir, 'models/siamese')
dir_tr = join(data_dir, 'train_{:d}_{:d}'.format(IMGS_DIM_2D[0], MAX_CLASSES))
dir_val = join(data_dir, 'val_{:d}_{:d}'.format(IMGS_DIM_2D[0], MAX_CLASSES))
dir_te = join(data_dir, 'test_{:d}_{:d}'.format(IMGS_DIM_2D[0], MAX_CLASSES))

def generators(dir_tr, dir_val, dir_te, batch_size, num_per_cls=1, num_per_cls_val=None):

	gen_tr = PairsImageDataGenerator()
	gen_val = PairsImageDataGenerator()
	gen_te = PairsImageDataGenerator()

	# sample = np.array(apply_to_images_in_subdirs(dir_tr, load_img_arr, num_per_cls=num_per_cls))
	# gen_tr.fit(sample)
	# gen_val.fit(sample)
	# gen_te.fit(sample)

	gen_tr = gen_tr.flow_from_directory(dir_tr, batch_size=batch_size)
	gen_val = gen_val.flow_from_directory(dir_val, batch_size=batch_size, num_per_cls=num_per_cls_val)
	gen_te = gen_te.flow_from_directory(dir_te, batch_size=batch_size, is_test=True)

	return gen_tr, gen_val, gen_te
	
class PairsImageDataGenerator(ImageDataGenerator):

	def __init__(self, *args, **kwargs):
		super(PairsImageDataGenerator, self).__init__(*args, **kwargs)

	def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
			 save_to_dir=None, save_prefix='', save_format='jpg', num_per_cls=None):
		raise NotImplementedError

	def flow_from_directory(
			self, dir_, target_size=IMGS_DIM_2D, color_mode='rgb',
			classes=None, class_mode='categorical', batch_size=32,
			shuffle=True, seed=None, save_to_dir=None, save_prefix='',
			save_format='jpg', num_per_cls=None, is_test=False):

		return PairsDirectoryIterator(dir_, self, batch_size, num_per_cls, is_test)

def imgreader(fns):
	return load_img_arr(fns[0]), load_img_arr(fns[1])

class PairsDirectoryIterator(object):

	def __init__(self, dir_, image_data_generator, batch_size=32, num_per_cls=None, is_test=False, use_numpy=False, save_as_npy=False):

		self.dir_ = dir_
		# self.paths, self.y = self._get_paths_labels_from_dir(dir_, num_per_cls, save_as_npy=save_as_npy)
		self.X, self.paths, self.y = self._get_paths_labels_from_dir(dir_, num_per_cls, use_numpy=use_numpy, save_as_npy=save_as_npy)
		self.batch_size = batch_size
		self._init_data_generator()
		self.image_data_generator = image_data_generator
		self.lock = threading.Lock()
		if not use_numpy:
			self.pool = Pool(processes=8)
		self.use_numpy = use_numpy
		self.is_test = is_test

	@staticmethod
	def _get_paths_labels_from_dir(dir_, num_per_cls, use_numpy=False, save_as_npy=False):
		def path_label(p): return [p, basename(dirname(p))]
		paths_labels = apply_to_images_in_subdirs(dir_, path_label, num_per_cls=num_per_cls)
		paths_labels = np.array(paths_labels)

		paths = paths_labels[:, 0]
		labels = paths_labels[:, 1].astype(int)

		X = []

		if save_as_npy:
			X = np.zeros((len(labels), IMGS_DIM_3D[0], IMGS_DIM_3D[1], IMGS_DIM_3D[2]))
			for i, path in enumerate(paths.tolist()):
				X[i, :] = load_img_arr(path)
			dir_name = dir_.split('/')[-1]
			np.save(join(data_dir, dir_name + '_X.npy'), X)
			np.save(join(data_dir, dir_name + '_y.npy'), labels)
			np.save(join(data_dir, dir_name + '_paths.npy'), paths)
			print "Saved %d %s images as numpy" % (X.shape[0], dir_name)
		if use_numpy:
			dir_name = dir_.split('/')[-1]
			X = np.load(join(data_dir, dir_name + '_X.npy'))
			labels = np.load(join(data_dir, dir_name + '_y.npy'))
			paths = np.load(join(data_dir, dir_name + '_paths.npy'))			
		
		return X, paths, labels
		
	def _init_data_generator(self):
		self.data_generator = pairs_generator(self.paths, self.y, self.batch_size)

	def iter(self):
		return self

	def next(self):

		debug = False

		if debug: start_time = time.time()
		with self.lock:
			try:
				indices_batch, paths_batch, y_batch = next(self.data_generator)
			except StopIteration:
				self._init_data_generator()
				indices_batch, paths_batch, y_batch = next(self.data_generator)

		if self.use_numpy and not self.is_test:
			X_batch = np.zeros((self.batch_size, 2, IMGS_DIM_3D[0], IMGS_DIM_3D[1], IMGS_DIM_3D[2]))
			for i in range(len(indices_batch)):
				X_batch[i, 0, :] = self.X[indices_batch[i][0], :]
				X_batch[i, 1, :] = self.X[indices_batch[i][1], :]
			X_batch = X_batch/float(IMGS_DIM_2D[0])
			X_batch = [X_batch[:, 0], X_batch[:, 1]]
		else:
			X_batch = self.pool.map(imgreader, paths_batch)
			X_batch = np.array(X_batch)/float(IMGS_DIM_2D[0])
			X_batch = [X_batch[:, 0], X_batch[:, 1]]

		if self.is_test:
			return y_batch, paths_batch
		else:
			if debug: print time.time()-start_time
			return X_batch, y_batch

def pairs_generator(paths, y, batch_size):
	
	pairs = generate_pairs(y, alternate=True)
	
	while True:
		indices_batch, paths_batch, y_batch = [], [], []
		
		for i in range(batch_size):
			
			try:
				pair = next(pairs)

			except StopIteration:
				return
			
			index_a, index_b = pair
			path_a, path_b = paths[index_a], paths[index_b]
			class_a, class_b = y[index_a], y[index_b]     
			
			indices_batch.append(pair)
			paths_batch.append([path_a, path_b])
			y_batch.append(float(class_a != class_b))
					
		paths_batch = np.array(paths_batch)

		y_batch = np.array(y_batch)
					
		yield indices_batch, paths_batch, y_batch

def generate_pairs(y, alternate=False):

	indices = range(len(y))

	if not alternate:

		return combinations(indices, 2)

	else:

		in_class_pairs = []
		for class_ in set(y):
			in_class_pairs.append(itertools.combinations(np.where(y==class_)[0].tolist(), 2))
		positive_pairs = roundrobin(*in_class_pairs)

		negative_pairs = combinations(indices, 2)

		return itertools.chain(*itertools.izip(positive_pairs, negative_pairs))

def roundrobin(*iterables):
	pending = len(iterables)
	nexts = cycle(it.next for it in iterables)
	while pending:
		try:
			for next in nexts:
				yield next()
		except StopIteration:
			pending -= 1
			nexts = cycle(islice(nexts, pending))

def apply_to_images_in_subdirs(parent_dir, func, num_per_cls=None, **kwargs):
	results = []
	for cls_dir_name in listdir(parent_dir):
		cls_dir = abspath(join(parent_dir, cls_dir_name))
		r = _apply_to_first_n_in_dir(func, cls_dir, num_per_cls, **kwargs)
		results += r
	return results

def _apply_to_first_n_in_dir(func, dir_, num_per_cls, **kwargs):
	if not isdir(dir_):
		return []
	results = []
	for path in listdir(dir_)[:num_per_cls]:
		abspath_ = abspath(join(dir_, path))
		result = func(abspath_, **kwargs)
		results.append(result)
	return results

def build_siamese_net(model_fname, metrics=None):
	
	processor = _shared_net(model_fname)

	input_1 = Input(shape=IMGS_DIM_3D)
	input_2 = Input(shape=IMGS_DIM_3D)

	processed_1 = processor(input_1)
	processed_2 = processor(input_2)

	distance = Lambda(_euclidean_distance, output_shape=_eucl_dist_output_shape)([processed_1, processed_2])

	model = Model(input=[input_1, input_2], output=distance)
	
	model.compile(loss=_contrastive_loss, optimizer='rmsprop', metrics=metrics)

	return model

def train_model(model, model_fname, gen_tr, gen_val):
			
	hist = model.fit_generator(
		generator=gen_tr,
		nb_epoch=MAX_EPOCHS,
		samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
		validation_data=gen_val,
		nb_val_samples=VAL_SAMPLES_PER_EPOCH,
		callbacks=[ModelCheckpoint(model_fname + '.checkpoint', save_best_only=True)],
		verbose=2
	)
	
	return hist
	
def _shared_net(model_fname):
		
	nb_filters = 32
	nb_pool = 2
	nb_conv = 3

	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,  activation='relu', input_shape=IMGS_DIM_3D))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(p=0.1))

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(p=0.1))

	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(p=0.1))
	model.add(Dense(512, activation='relu'))
	
	plot(model, to_file=model_fname + '.base_net.png')

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

def save_model(model_fname, model, hist):
	arch_fname = model_fname + '.arch.json'
	weights_fname = model_fname + '.weights.h5'
	hist_fname = model_fname + '.hist.npy'
	open(arch_fname, 'w').write(model.to_json())
	model.save_weights(weights_fname, overwrite=True)
	np.save(hist_fname, hist)

def load_model(model_fname):
	arch_fname = model_fname + '.arch.json'
	weights_fname = model_fname + '.weights.h5'
	hist_fname = model_fname + '.hist.npy'
	model_json_string = open(arch_fname).read()
	model=model_from_json(model_json_string)
	model.load_weights(weights_fname)
#	hist = np.load(hist_fname)
	hist = []
	return model, hist

def generate_test(gen_te):

	y_test = []
	path_pairs = []
	while True:
		data = next(gen_te)
		if data:
			y_batch, pairs = data
			y_test += list(y_batch)
			path_pairs += list(pairs)
		else:
			break
		if len(y_test) > TOTAL_TEST_SAMPLES:
			break
	return np.array(y_test), np.array(path_pairs)

def test_model(model, gen_te, metrics, regenerate_test=False):

	if regenerate_test:
		y_test, path_pairs = generate_test(gen_te)
		np.save(join(models_dir, 'y_test_64'), y_test)
		np.save(join(models_dir, 'test_pairs_64'), path_pairs)
		print "Generated test"
	else:
		y_test = np.load(join(models_dir, 'y_test_64.npy'))
		path_pairs = np.load(join(models_dir, 'test_pairs_64.npy'))
		print "Loaded test"

	y_test_pred = []
	perf = {}
	pool = Pool(processes=4)

	batch_size = BATCH_SIZE
	num_batches = path_pairs.shape[0]/batch_size
	test_batches = num_batches

	for i in range(test_batches):
		print "Test batch : {0}".format(i)
		batch = range(i * batch_size, (i+1) * batch_size)
		pairs_batch = path_pairs[batch, :]
		X_batch = pool.map(imgreader, pairs_batch)
		X_batch = np.array(X_batch)
		X_batch = [X_batch[:, 0], X_batch[:, 1]]
		y_test_pred += list(model.predict(X_batch))

	y_test = y_test[:test_batches*batch_size]

	perfs = []
	for comp_metric in metrics:
		perfs.append(comp_metric(y_test, y_test_pred))

	return y_test, y_test_pred, perfs

def comp_f1(y_true, y_pred):
	print y_true
	y_pred = [int(y.ravel() > 0.5) for y in y_pred]
	print y_pred
	return f1_score(y_true, y_pred)

def comp_precision(y_true, y_pred):
	y_pred = [int(y.ravel() > 0.5) for y in y_pred]
	return precision_score(y_true, y_pred)

def comp_recall(y_true, y_pred):
	y_pred = [int(y.ravel() > 0.5) for y in y_pred]
	return recall_score(y_true, y_pred)

def comp_accuracy(y_true, y_pred):
	y_pred = [int(y.ravel() > 0.5) for y in y_pred]
	return accuracy_score(y_true, y_pred)

def comp_roc_auc(y_true, y_pred):
	y_pred = [int(y.ravel() > 0.5) for y in y_pred]
	return roc_auc_score(y_true, y_pred)

BATCH_SIZE = 32

MAX_EPOCHS = 20
TRAIN_SAMPLES_PER_EPOCH = 5120
VAL_SAMPLES_PER_EPOCH = 512
TOTAL_TEST_SAMPLES = 5008

NUM_PER_CLS = 1
MARGIN = 0.1

rebuild_model= False
resave_model= True
regenerate_test = False
save_test_results = False

gen_tr, gen_val, gen_te = generators(dir_tr, dir_val, dir_te, batch_size=BATCH_SIZE, num_per_cls=NUM_PER_CLS)
print "\nBuilt generators"

model_fname = join(models_dir, 'siamese_64')
test_metrics = [comp_f1, comp_precision, comp_recall, comp_accuracy, comp_roc_auc]

if rebuild_model:
	model = build_siamese_net(model_fname)
	print "Built model"
	hist = train_model(model, model_fname, gen_tr, gen_val)
	print "Trained model"
	if resave_model:
		save_model(model_fname, model, hist)
else: 
	model, hist = load_model(model_fname)
	print "Loaded trained model"

y_test, y_test_pred, test_perf = test_model(model, gen_te, test_metrics, regenerate_test=regenerate_test)

if save_test_results:
	np.save(model_fname + '.y_test_pred', y_test_pred)
	np.save(model_fname + '.test_perf', test_perf)

print test_perf
