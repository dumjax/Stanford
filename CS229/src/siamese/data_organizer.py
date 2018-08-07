
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
from sklearn.model_selection import train_test_split
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

# In[2]:

IMGS_DIM_3D = (3, 64, 64)
IMGS_DIM_2D = IMGS_DIM_3D[1:]

DATA_DIR = '/data/paintersbynumbers/'

TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_INFO_FILE = join(DATA_DIR, 'train_info.csv')

NEW_TRAIN_DIR = join(DATA_DIR, 'train_{:d}'.format(IMGS_DIM_2D[0]))
NEW_VAL_DIR = join(DATA_DIR, 'val_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(DATA_DIR, 'test_{:d}'.format(IMGS_DIM_2D[0]))
		
VAL_SIZE = 0.1
TEST_SIZE = 0.1

MAX_CLASSES = 100

if MAX_CLASSES < 1584:
	NEW_TRAIN_DIR = NEW_TRAIN_DIR + '_{:d}'.format(MAX_CLASSES)
	NEW_VAL_DIR = NEW_VAL_DIR + '_{:d}'.format(MAX_CLASSES)
	NEW_TEST_DIR = NEW_TEST_DIR + '_{:d}'.format(MAX_CLASSES)

# def _organize_train_dir():
# 	paths, labels = _load_paths_labels_from_train_dir()
# 	labels_tr, labels_te, paths_tr, paths_te = train_test_split(labels, paths, test_size=TEST_SIZE, random_state=42)
# 	labels_tr, labels_val, paths_tr, paths_val = train_test_split(labels_tr, paths_tr, test_size=VAL_SIZE, random_state=42)

# 	classes = sorted(list(set(labels)))

# 	_save_images_to_dir(NEW_TRAIN_DIR, paths_tr, labels_tr, classes)
# 	print "Saved train images"
# 	_save_images_to_dir(NEW_VAL_DIR, paths_val, labels_val, classes)
# 	print "Saved val images"
# 	_save_images_to_dir(NEW_TEST_DIR, paths_te, labels_te, classes)
# 	print "Saved test images"

def _organize_train_dir():
    paths, labels = _load_paths_labels_from_train_dir()
    ind_tr, ind_val, ind_te, classes = _train_val_test_split_indices(labels, paths)

    _save_images_to_dir(NEW_TRAIN_DIR, paths[ind_tr], labels[ind_tr], classes)
    print "Saved train images"
    _save_images_to_dir(NEW_VAL_DIR, paths[ind_val], labels[ind_val], classes)
    print "Saved val images"
    _save_images_to_dir(NEW_TEST_DIR, paths[ind_te], labels[ind_te], classes)
    print "Saved test images"

def _load_paths_labels_from_train_dir():
	labels_lookup = load_train_info()
	paths, labels = [], []
	for name in listdir(TRAIN_DIR):
		abspath_ = abspath(join(TRAIN_DIR, name))
		paths.append(abspath_)
		labels.append(labels_lookup[name])

	return np.array(paths), LabelEncoder().fit_transform(labels)

def load_train_info():
	train_info = read_lines(TRAIN_INFO_FILE)[1:]
	parsed_train_info = {}
	# filename,artist,title,style,genre,date
	for l in train_info:
		split = l.split(',')
		parsed_train_info[split[0]] = split[1]
	return parsed_train_info

# def _train_val_test_split_indices(labels, paths=None):

#     split = StratifiedShuffleSplit(labels, n_iter=1, test_size=TEST_SIZE, random_state=42)
#     indices_tr, indices_val = next(iter(split))

#     return indices_tr, indices_val, [], split.classes

def _train_val_test_split_indices(labels, paths=None):

    split = StratifiedShuffleSplit(labels, n_iter=1, test_size=TEST_SIZE, random_state=42)
    indices_tr, indices_te = next(iter(split))

    labels = labels[indices_tr]
    split = StratifiedShuffleSplit(labels, n_iter=1, test_size=VAL_SIZE, random_state=42)    
    indices_tr_tr, indices_tr_val = next(iter(split))

    indices_val = indices_tr[indices_tr_val]
    indices_tr = indices_tr[indices_tr_tr]

    return indices_tr, indices_val, indices_te, split.classes

def _save_images_to_dir(dest_dir, src_paths, labels, distinct_classes):

	_make_dir_tree(dest_dir, distinct_classes)

	for src_path, label in zip(src_paths, labels):

		if MAX_CLASSES and label < MAX_CLASSES:

			dest_path = join(join(dest_dir, str(label)), basename(src_path))
			scaled_cropped_image = _save_scaled_cropped_img(src_path, dest_path)
				
def _make_dir_tree(dir_, classes):
	mkdir(dir_)
	for class_ in classes:

		if MAX_CLASSES and class_ < MAX_CLASSES:

			class_dir = join(dir_, str(class_))
			mkdir(class_dir)
		
def _save_scaled_cropped_img(src, dest):
	image = load_img(src)
	image = fit(image, IMGS_DIM_2D, method=LANCZOS)
	image.save(dest)
	return image

_organize_train_dir()