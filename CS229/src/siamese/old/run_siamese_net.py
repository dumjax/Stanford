
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

# In[2]:

IMGS_DIM_3D = (3, 64, 64)
IMGS_DIM_2D = IMGS_DIM_3D[1:]

DATA_DIR = '/data/paintersbynumbers/'

TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_INFO_FILE = join(DATA_DIR, 'train_info.csv')
TEST_X_FILE = join(DATA_DIR, 'submission_info.csv')
TEST_Y_FILE = join(DATA_DIR, 'solution_painter.csv')

ORGANIZED_DATA_INFO_FILE = 'organized_data_info_.json'
MODELS_DIR = join(DATA_DIR, 'models/siamese')
NEW_TRAIN_DIR = join(DATA_DIR, 'train_{:d}'.format(IMGS_DIM_2D[0]))
NEW_VAL_DIR = join(DATA_DIR, 'val_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(DATA_DIR, 'test_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(NEW_TEST_DIR, 'all')
NEW_TEST_GEN_DIR = join(DATA_DIR, 'test_{:d}_gen'.format(IMGS_DIM_2D[0]))
        
VAL_SIZE = 0.1


# In[3]:

# _organize_train_dir()
# _organize_test_dir()

def _organize_train_dir():
    paths, labels = _load_paths_labels_from_train_dir()
    ind_tr, ind_val, classes = _train_val_split_indices(labels)
    _save_images_to_dir(NEW_TRAIN_DIR, paths[ind_tr], labels[ind_tr], classes)
    _save_images_to_dir(NEW_VAL_DIR, paths[ind_val], labels[ind_val], classes)
    _save_organized_data_info(classes, ind_tr, ind_val)  

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

def _train_val_split_indices(labels):
    split = StratifiedShuffleSplit(labels, n_iter=1, test_size=VAL_SIZE, random_state=42)
    indices_tr, indices_val = next(iter(split))
    
    return indices_tr, indices_val, split.classes

def _save_organized_data_info(classes, indices_tr, indices_val):
    info = {
        'dir_tr': NEW_TRAIN_DIR,
        'num_tr': len(indices_tr),
        'dir_val': NEW_VAL_DIR,
        'num_val': len(indices_val),
        'num_distinct_cls': len(classes),
    }
    save_organized_data_info(info, IMGS_DIM_2D[0])

def save_organized_data_info(info, imgs_dim_1d):
    with open(_organized_data_info_file_dim(imgs_dim_1d), 'w') as f:
        dump(info, f)

def load_organized_data_info(imgs_dim_1d):
    with open(_organized_data_info_file_dim(imgs_dim_1d), 'r') as f:
        return load(f)
    
def _organized_data_info_file_dim(imgs_dim_1d):
    split = ORGANIZED_DATA_INFO_FILE.split('.')
    split[0] += str(imgs_dim_1d)
    return join(DATA_DIR, '.'.join(split))

def _save_images_to_dir(dest_dir, src_paths, labels, distinct_classes):

    # _make_dir_tree(dest_dir, distinct_classes)

    for src_path, label in zip(src_paths, labels):
        dest_path = join(join(dest_dir, str(label)), basename(src_path))
        # scaled_cropped_image = _save_scaled_cropped_img(src_path, dest_path)
            
def _make_dir_tree(dir_, classes):
    mkdir(dir_)
    for class_ in classes:
        class_dir = join(dir_, str(class_))
        mkdir(class_dir)
        
def _save_scaled_cropped_img(src, dest):
    image = load_img(src)
    image = fit(image, IMGS_DIM_2D, method=LANCZOS)
    image.save(dest)
    return image

def _organize_test_dir():
    # makedirs(NEW_TEST_DIR)

    num_test_samples = 0
    for name in listdir(TEST_DIR):
        src_path = abspath(join(TEST_DIR, name))
        dest_path = join(NEW_TEST_DIR, name)
        try:
            # _save_scaled_cropped_img(src_path, dest_path)
            num_test_samples += 1
        except IOError:
            pass
    _append_num_te_to_organized_data_info(num_test_samples)
    
def _append_num_te_to_organized_data_info(num_test_samples):
    data_info = load_organized_data_info(IMGS_DIM_2D[0])
    data_info['dir_te'] = dirname(NEW_TEST_DIR)
    data_info['num_te'] = num_test_samples
    save_organized_data_info(data_info, IMGS_DIM_2D[0])


# In[4]:

_organize_train_dir()

# _organize_test_dir()

# Data provider stack

# In[5]:

def generators(dir_tr, dir_val, dir_te, batch_size, num_samples_per_cls=1, num_samples_per_cls_val=None):

    gen_tr = PairsImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    gen_val = PairsImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    sample = np.array(apply_to_images_in_subdirs(dir_tr, load_img_arr, num_samples_per_cls=num_samples_per_cls))
    gen_tr.fit(sample)
    gen_val.fit(sample)

    gen_tr = gen_tr.flow_from_directory(dir_tr, batch_size=batch_size)
    
    gen_val = gen_val.flow_from_directory(dir_val, batch_size=batch_size, num_samples_per_cls=num_samples_per_cls_val)
    
    if dir_te is not None:
        gen_te = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        gen_te.fit(sample)
        num_test_samples = load_organized_data_info(IMGS_DIM_2D[0])['num_te']
        batch_size=18
        gen_te = gen_te.flow_from_directory(dir_te, class_mode=None, batch_size=batch_size, 
                                            target_size=IMGS_DIM_2D, shuffle=False, save_format='jpg',
                                           save_to_dir=NEW_TEST_GEN_DIR, save_prefix='aug')   
        num_runs = 0
        for X in gen_te:
            num_runs+=1
            if num_runs == num_test_samples/batch_size:
                break
                
        for fn in os.listdir(NEW_TEST_GEN_DIR):
            fn = join(NEW_TEST_GEN_DIR, fn)
            if os.path.isfile(fn):
                new_fn = join(NEW_TEST_GEN_DIR, fn.split('_')[-1])
                os.rename(fn, new_fn)

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
            save_format='jpg', num_samples_per_cls=None):

        return PairsDirectoryIterator(
            dir_, self, batch_size, num_samples_per_cls)


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
                 batch_size=32, num_samples_per_cls=None):
    
        self.dir_ = dir_
        self.paths, self.y = self._get_paths_labels_from_dir(dir_, num_samples_per_cls)
        self.batch_size = batch_size
        self._init_pairs_generator()
        self.image_data_generator = image_data_generator
        self.lock = threading.Lock()
        self.pool = Pool(processes=8)

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
        # X_batch = self.pool.map(imgreader_pool_wrapper, [(paths_pair, self.image_data_generator) for paths_pair in paths_batch])
        X_batch = np.array(X_batch)
        X_batch = [X_batch[:, 0], X_batch[:, 1]]

        # X_batch = []
        # for path_a, path_b in paths_batch:
        #     image_a, image_b = load_img_arr(path_a), load_img_arr(path_b)
        #     image_a = self._std_random_transform_img(image_a)
        #     image_b = self._std_random_transform_img(image_b)
        #     X_batch.append([image_a, image_b])
        # X_batch = np.array(X_batch)

        print "Processed batch: {0}".format(time.time()-start)

        return X_batch, y_batch

    # def _std_random_transform_img(self, img):
    #     img = self.image_data_generator.random_transform(img)
    #     return self.image_data_generator.standardize(img)


def pairs_generator(X, y, batch_size, pair_func):
    
    hard_positive_mining=True
    
    singles = range(y.shape[0])
    shuffle(singles)
    pairs = combinations(singles, 2)
    
    while True:
        X_batch, y_batch = [], []
        
        for i in range(batch_size):
            try:
                pair_indices = next(pairs)
            except StopIteration:
                return
            index_a, index_b = int(pair_indices[0]), int(pair_indices[1])
            X_batch.append(pair_func(X[index_a], X[index_b]))
            y_batch.append(int(y[index_a] != y[index_b]))
            
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
            
        yield X_batch, y_batch

def alt_pairs_generator(dir_, batch_size, pair_func):
        
    data_info = load_organized_data_info(IMGS_DIM_2D[0])
    classes = range(data_info['num_distinct_cls'])
    
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


# In[6]:

BATCH_SIZE = 16

MAX_EPOCHS = 20
TRAIN_SAMPLES_PER_EPOCH = 1000
VAL_SAMPLES_PER_EPOCH = 100
TEST_SAMPLES_TOTAL = 10000

# MAX_EPOCHS = 100
# TRAIN_SAMPLES_PER_EPOCH = data_info['num_tr']
# VAL_SAMPLES_PER_EPOCH = data_info['num_val']
# TEST_SAMPLES_TOTAL = data_info['num_te']/100

NUM_SAMPLES_PER_CLS = 1

MARGIN = 1

data_info = load_organized_data_info(IMGS_DIM_2D[0])
dir_tr = data_info['dir_tr']
dir_val = data_info['dir_val']
dir_te = data_info['dir_te']

def build_siamese_net():
    
    print "Building siamese net"

    processor = _shared_net(full=False)

    input_1 = Input(shape=IMGS_DIM_3D)
    input_2 = Input(shape=IMGS_DIM_3D)

    processed_1 = processor(input_1)
    processed_2 = processor(input_2)

    distance = Lambda(_euclidean_distance, output_shape=_eucl_dist_output_shape)([processed_1, processed_2])

    model = Model(input=[input_1, input_2], output=distance)
    
    rms = RMSprop(lr=1e-5)
    model.compile(loss=_contrastive_loss, optimizer='rmsprop')

    return model

def train_model():
        
    print "Training model"
    
    hist = model.fit_generator(
        generator=gen_tr,
        nb_epoch=MAX_EPOCHS,
        samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
        validation_data=gen_val,
        nb_val_samples=VAL_SAMPLES_PER_EPOCH,
        callbacks=[ModelCheckpoint(model_fname + '_checkpoint', save_best_only=True)],
        verbose=2
    )
    
    return hist
    
def _shared_net(full=False):
        
    print "Building shared net"

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
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(p=0.1))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(mode=2))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization(mode=2))     
    
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

def test_model():

    y_true = []
    y_pred = []
    test_indices = []
    
    num_tests = 0
    batch_size=32

    pool = Pool(processes=8)
    for batch in read_lines_in_batches(TEST_X_FILE, batch_size):

        print "Num_tests: {0}\r".format(float(num_tests)/TEST_SAMPLES_TOTAL), 

        imgs = [(join(NEW_TEST_GEN_DIR, line[1]), join(NEW_TEST_GEN_DIR, line[2])) for line in batch 
                if os.path.isfile(join(NEW_TEST_GEN_DIR, line[1])) and os.path.isfile(join(NEW_TEST_GEN_DIR, line[2]))]
        X_batch=pool.map(imgreader, imgs)

        for line in batch:
            ind = int(line[0])
            test_indices.append(ind)

            if os.path.isfile(img_1) and os.path.isfile(img_2):
                X_batch.append([x_1, x_2])
                y_true.append(test_labels[ind])

        # for line in batch:
        #     ind = int(line[0])
        #     test_indices.append(ind)
        #     img_1 = join(NEW_TEST_GEN_DIR, line[1])
        #     img_2 = join(NEW_TEST_GEN_DIR, line[2])

        #     if os.path.isfile(img_1) and os.path.isfile(img_2):
        #         x_1 = load_img_arr(img_1)
        #         x_2 = load_img_arr(img_2)
        #         X_batch.append([x_1, x_2])
        #         y_true.append(test_labels[ind])

        X_batch = np.array(X_batch)
        X_batch = [X_batch[:,0], X_batch[:,1]]
        y_pred_batch = list(model.predict(X_batch, batch_size=batch_size))

        y_pred += y_pred_batch

        num_tests = len(y_true)
        if num_tests > TEST_SAMPLES_TOTAL:
            break

    y_true = np.array(y_true)
    print y_true, y_pred
    y_pred = np.array([y[0] for y in y_pred])

    print y_true, y_pred
    print compute_accuracy(y_pred, y_true)
    return

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


# Build and train siamese net

# In[ ]:

rebuild=True
model_fname = join(MODELS_DIR, 'siamese_m1')
if rebuild:
    gen_tr, gen_val = generators(dir_tr, dir_val, None, batch_size=BATCH_SIZE, num_samples_per_cls=NUM_SAMPLES_PER_CLS)
    print "Build generators"
    model = build_siamese_net()
    hist = train_model()
    print "Trained models"
    save_model(model)
else: 
    model = load_model()
    print "Loaded model"

# # In[ ]:

test_labels = {}
batch_size=1
for batch in read_lines_in_batches(TEST_Y_FILE, batch_size):
    for line in batch:
        test_labels[int(line[0])] = 1-int(float(line[1]))
print "Loaded test labels"

# # Test siamese net

# # In[ ]:

test_model()
