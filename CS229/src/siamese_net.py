
# coding: utf-8

# Data organizer stack

# In[ ]:

from os import mkdir, listdir, makedirs
from os.path import join, abspath, basename, splitext, dirname, isdir

import numpy as np
np.random.seed(42)

from json import load, dump
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from PIL.Image import LANCZOS
from PIL.ImageOps import fit

from keras.preprocessing.image import load_img, ImageDataGenerator

from utils import read_lines, load_img_arr

IMGS_DIM_2D = (256, 256)
VAL_SIZE = 0.1

DATA_DIR = '/data/paintersbynumbers/'

TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_INFO_FILE = join(DATA_DIR, 'train_info.csv')

ORGANIZED_DATA_INFO_FILE = 'organized_data_info_.json'
MODELS_DIR = join(DATA_DIR, 'models')
NEW_TRAIN_DIR = join(DATA_DIR, 'train_{:d}'.format(IMGS_DIM_2D[0]))
NEW_VAL_DIR = join(DATA_DIR, 'val_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(DATA_DIR, 'test_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(NEW_TEST_DIR, 'all')

def _organize_train_dir():
    paths, labels = _load_paths_labels_from_train_dir()
    ind_tr, ind_val, classes = _train_val_split_indices(labels)
    _save_images_to_dir(NEW_TRAIN_DIR, paths[ind_tr], labels[ind_tr], classes)
    _save_images_to_dir(NEW_VAL_DIR, paths[ind_val], labels[ind_val], classes)
    
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

    _save_organized_data_info(split.classes, indices_tr, indices_val)
    
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

    _make_dir_tree(dest_dir, distinct_classes)

    for src_path, label in zip(src_paths, labels):
        dest_path = join(join(dest_dir, str(label)), basename(src_path))
        scaled_cropped_image = _save_scaled_cropped_img(src_path, dest_path)
        
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
    makedirs(NEW_TEST_DIR)

    num_test_samples = 0
    for name in listdir(TEST_DIR):
        src_path = abspath(join(TEST_DIR, name))
        dest_path = join(NEW_TEST_DIR, name)
        try:
            _save_scaled_cropped_img(src_path, dest_path)
            num_test_samples += 1
        except IOError:
            pass

    _append_num_te_to_organized_data_info(num_test_samples)
    
def _append_num_te_to_organized_data_info(num_test_samples):
    data_info = load_organized_data_info(IMGS_DIM_2D[0])
    data_info['dir_te'] = dirname(NEW_TEST_DIR)
    data_info['num_te'] = num_test_samples
    save_organized_data_info(data_info, IMGS_DIM_2D[0])
    

def init_directory_generator(
        gen, dir_, batch_size, target_size=IMGS_DIM_2D,
        class_mode='categorical', shuffle_=True):

    return gen.flow_from_directory(
        dir_,
        class_mode=class_mode,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=shuffle_)


# In[ ]:

# _organize_train_dir()
# _organize_test_dir()


# Data provider stack

# In[ ]:

import threading
from random import shuffle
from itertools import combinations, chain
from math import ceil

def train_val_pairs_dirs_generators(
    batch_size, dir_tr, dir_val, num_groups_tr, num_groups_val, num_samples_per_cls=2, num_samples_per_cls_val=None):

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
    
    gen_val = PairsImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    sample = np.array(apply_to_images_in_subdirs(dir_tr, load_img_arr, num_samples_per_cls=num_samples_per_cls))
    gen_tr.fit(sample)
    gen_val.fit(sample)

    gen_tr = gen_tr.flow_from_directory(
        dir_tr, batch_size=batch_size, num_groups=num_groups_tr)
    
    gen_val = gen_val.flow_from_directory(
        dir_val, batch_size=batch_size, num_groups=num_groups_val,
        num_samples_per_cls=num_samples_per_cls_val)
    
    return gen_tr, gen_val

class PairsImageDataGenerator(ImageDataGenerator):

    def __init__(self, *args, **kwargs):
        super(PairsImageDataGenerator, self).__init__(*args, **kwargs)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg',
             num_groups=43, num_samples_per_cls=None):
        raise NotImplementedError

    def flow_from_directory(
            self, dir_, target_size=(256, 256), color_mode='rgb',
            classes=None, class_mode='categorical', batch_size=32,
            shuffle=True, seed=None, save_to_dir=None, save_prefix='',
            save_format='jpeg', num_groups=43, num_samples_per_cls=None):

        return PairsDirectoryIterator(
            dir_, num_groups, self, batch_size, num_samples_per_cls)

class PairsDirectoryIterator(object):

    def __init__(self, dir_, num_groups, image_data_generator,
                 batch_size=32, num_samples_per_cls=None):

        paths, y = self._get_paths_labels_from_dir(dir_, num_samples_per_cls)
        self.paths = paths
        self.y = y
        self.num_groups = num_groups
        self.batch_size = batch_size
        self._init_pairs_generator()
        self.image_data_generator = image_data_generator
        self.lock = threading.Lock()

    @staticmethod
    def _get_paths_labels_from_dir(dir_, num_per_cls):
        def path_label(p): return [p, basename(dirname(p))]
        paths_labels = apply_to_images_in_subdirs(dir_, path_label, num_per_cls)
        paths_labels = np.array(paths_labels)
        return paths_labels[:, 0], paths_labels[:, 1].astype(int)

    def _init_pairs_generator(self):
        self.pairs_generator = pairs_generator(
            self.paths, self.y, self.batch_size,
            lambda a, b: [a, b], self.num_groups)

    def iter(self):
        return self

    def next(self):
        with self.lock:
            try:
                paths_batch, y_batch = next(self.pairs_generator)
            except StopIteration:

                self._init_pairs_generator()
                paths_batch, y_batch = next(self.pairs_generator)

        X_batch = []
        for path_a, path_b in paths_batch:
            image_a, image_b = load_img_arr(path_a), load_img_arr(path_b)
            image_a = self._std_random_transform_img(image_a)
            image_b = self._std_random_transform_img(image_b)
            X_batch.append([image_a, image_b])
        X_batch = np.array(X_batch)

        return [X_batch[:, 0], X_batch[:, 1]], y_batch

    def _std_random_transform_img(self, img):
        img = self.image_data_generator.random_transform(img)
        return self.image_data_generator.standardize(img)
    
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

def pairs_generator(X, y, batch_size, pair_func, num_groups):
    grouped_indices = _split_into_groups(y, num_groups)
    merged_combinations = _merge_within_groups_combinations(grouped_indices)

    while True:
        X_batch, y_batch = [], []

        for _ in range(batch_size):
            try:
                pair_indices = next(merged_combinations)
            except StopIteration:
                return

            index_a, index_b = int(pair_indices[0]), int(pair_indices[1])
            X_batch.append(pair_func(X[index_a], X[index_b]))
            y_batch.append(int(y[index_a] == y[index_b]))

        yield np.array(X_batch), np.array(y_batch)

def _split_into_groups(y, num_groups):
    groups = [[] for _ in range(num_groups)]
    group_index = 0

    for cls in set(y):
        this_cls_indices = np.where(y == cls)[0]
        num_cls_samples = len(this_cls_indices)

#         num_cls_split_groups = ceil(num_cls_samples / 500)
        num_cls_split_groups = max(1, ceil(num_cls_samples / 500))
        split = np.array_split(this_cls_indices, num_cls_split_groups)

        for cls_group in split:
            groups[group_index] = np.hstack((groups[group_index], cls_group))
            group_index = (group_index + 1) % num_groups

    return groups

def _merge_within_groups_combinations(grouped_indices):
    for gi in grouped_indices:
        shuffle(gi)
    group_combinations = [combinations(gi, 2) for gi in grouped_indices]
    shuffle(group_combinations)
    return chain.from_iterable(group_combinations)

def test_generator(dir_tr, num_samples_per_cls=2):
    gen_te = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    sample = apply_to_images_in_subdirs(dir_tr, load_img_arr, num_samples_per_cls=num_samples_per_cls)
    sample = np.array(sample)
    gen_te.fit(sample)
    return gen_te


# Build and train siamese net

# In[ ]:

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D 
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras import backend as K
K.set_image_dim_ordering('th')

SIAMESE_MODEL_FILE = join(MODELS_DIR, 'siamese.h5')

IMGS_DIM_3D = (3, 256, 256)
BATCH_SIZE = 32
MAX_EPOCHS = 20 #500

TRAIN_SAMPLES_PER_EPOCH = 10000
VAL_SAMPLES_PER_EPOCH = 1000

NUM_GROUPS_TR = NUM_GROUPS_VAL = 1
NUM_SAMPLES_PER_CLS = 2
        
data_info = load_organized_data_info(IMGS_DIM_3D[1])
dir_tr = data_info['dir_tr']
dir_val = data_info['dir_val']
    
def build_train_siamese_net():
    
    gen_tr, gen_val = train_val_pairs_dirs_generators(BATCH_SIZE, dir_tr, dir_val, 
                                                      NUM_GROUPS_TR, NUM_GROUPS_VAL, NUM_SAMPLES_PER_CLS)
    
    model = siamese_net()
    
    print "Training model"
    
    model.fit_generator(
        generator=gen_tr,
        nb_epoch=MAX_EPOCHS,
#         samples_per_epoch=data_info['num_tr'],
        samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
        validation_data=gen_val,
        #nb_val_samples=data_info['num_val'],
        nb_val_samples=VAL_SAMPLES_PER_EPOCH,
        callbacks=[ModelCheckpoint(SIAMESE_MODEL_FILE, save_best_only=True)],
        verbose=2
    )
    
    return model
    
def siamese_net():
    
    print "Building siamese net"
    
    processor = _shared_net()
    
    input_1 = Input(shape=IMGS_DIM_3D)
    input_2 = Input(shape=IMGS_DIM_3D)
    
    processed_1 = processor(input_1)
    processed_2 = processor(input_2)
    
    distance = Lambda(_euclidean_distance, output_shape=_eucl_dist_output_shape)([processed_1, processed_2])
    
    model = Model(input=[input_1, input_2], output=distance)
    
    model.compile(loss=_contrastive_loss, optimizer='rmsprop')
    
    return model
    
def _shared_net():
    
    print "Building shared net"
    
    # input image dimensions
#     img_colours, img_rows, img_cols = IMGS_DIM_3D

    # number of convolutional filters to use
    nb_filters = 32

    # size of pooling area for max pooling
    nb_pool = 2

    # convolution kernel size
    nb_conv = 3

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,  activation='relu',
                            input_shape=IMGS_DIM_3D, border_mode='valid'))
#     model.add(BatchNormalization(axis=1, mode=2))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
#     model.add(BatchNormalization(axis=1, mode=2))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(64, input_shape=IMGS_DIM_3D, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
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

    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def test(model):
    gen_te = test_generator(dir_tr=dir_tr, num_per_cls=NUM_SAMPLES_PER_CLS)
    pass


# In[ ]:

model = build_train_siamese_net()

# Test siamese net

# In[ ]:

# test(model)

