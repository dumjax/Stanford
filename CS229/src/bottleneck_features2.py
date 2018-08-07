import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import sys, os
sys.path.insert(0, "/home/ubuntu/cs229_project/deep-learning-models/")
from imagenet_utils import preprocess_input, decode_predictions
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint


import vgg16_new

nb_epoch = 100

def computeFeatures():
    # path to the model weights file.
    model_weights_path = '/data/paintersbynumbers/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # dimensions of our images.
    dims=(224, 224)
    data_dir='/data/paintersbynumbers/'
    train_data_dir = '/data/paintersbynumbers/train_256'
    validation_data_dir = data_dir+'data/validation'
    nb_train_samples = 78000
    nb_validation_samples = 800
    nb_epoch = 100

    total_images = 80000

    #img_dir = data_dir + 'train_256/'
    img_dir = data_dir + 'val_256/'
    features_dir = data_dir + 'features/'
    weights_dir = data_dir + 'weights/'
    weights_path = weights_dir + 'weights.h5'


    output_shape = 9

    model = vgg16_new.VGG16(model_weights_path)

    #featurizer = VGG16(include_top=False,weights='imagenet')

    X=list()
    Y=list()
    paths = os.listdir(img_dir)
    for fname in paths:
        if not fname.startswith('.'):
            subfolder = os.path.join(img_dir, fname)
            for img_file in os.listdir(subfolder):
                if not img_file.startswith('.'):
                    img = image.load_img(os.path.join(subfolder, img_file), target_size=dims)
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    #x = featurizer.predict(img)
                    x= model.predict(img)
                    y = int(fname)
                    try:
                        X
                    except NameError:
                        X = [x[0]]
                    else:
                        X.append(x[0])

                    try:
                        Y
                    except NameError:
                        Y = [y]
                    else:
                        Y.append(y)

                    print(float(len(X))/total_images)
    #Y = to_categorical(Y)
#	np.save(open(features_dir + 'features.npy', 'wb'), X)
#	np.save(open(features_dir + 'labels.npy', 'wb'), Y)

    return np.array(X) , np.array(Y)

from sklearn.cross_validation import train_test_split

def train_top_model():
    train_data = np.load('/data/paintersbynumbers/features/features_train.npy','r')
    train_labels = to_categorical(np.load('/data/paintersbynumbers/features/labels_train.npy','r'))
    #X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    val_data = np.load('/data/paintersbynumbers/features/features_val.npy','r')
    val_labels = to_categorical(np.load('/data/paintersbynumbers/features/labels_val.npy','r'))
    X_train=train_data
    y_train=train_labels
    X_val=val_data
    y_val=val_labels
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(y_train.shape[1], activation='sigmoid'))    
    model.add(Dense(y_train.shape[1], activation='softmax'))

    #model.compile( loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.compile( loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='/data/paintersbynumbers/weights/weights_topmodel.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    model.fit(X_train,y_train,
              nb_epoch=nb_epoch, batch_size=32,validation_data=(X_val, y_val))
    model.save_weights('/data/paintersbynumbers/models/categorical_weights.h5')


