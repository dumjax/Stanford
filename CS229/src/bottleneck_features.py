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
from keras.layers import Activation, BatchNormalization, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint


import vgg16_new

nb_epoch = 10
img_size = 64
def computeFeatures(filename):
    # path to the model weights file.
    model_weights_path = '/data/paintersbynumbers/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    #model_weights_path = '/data/paintersbynumbers/models/tmp/weights_64_finetuned.11-0.42.h5'

    data_dir='/data/paintersbynumbers/'
    #img_dir = data_dir + 'train_256/'
    img_dir = data_dir + filename+'/'
    features_dir = data_dir + 'features/'
    model = vgg16_new.VGG16(model_weights_path,img_size,True,False)

    X=list()
    Y=list()
    P = list()
    paths = os.listdir(img_dir)
    i=1
    for fname in paths:
        if not fname.startswith('.'):
            subfolder = os.path.join(img_dir, fname)
            for img_file in os.listdir(subfolder):
                if( not img_file.startswith('.')): # and int(fname)<101:
                    P.append(os.path.join(subfolder, img_file))
                    print('painter '+img_file+ '---' + str(i))
                    img = image.load_img(os.path.join(subfolder, img_file),target_size=(img_size,img_size))
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
                    i=i+1
                    # print(float(len(X)))
    #Y = to_categorical(Y)
#	np.save(features_dir + 'features_'+namefeatures+'.npy', X,allow_pickle=False)
#	np.save(features_dir + 'labels_'+namefeatures+'.npy', Y)

    return np.array(X) , np.array(Y), np.array(P)


def computePairs():

    model_weights_path = '/data/paintersbynumbers/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    top_model_weights_path = '/data/paintersbynumbers/models/basic_nn_64_2.h5'
    #fine-tuned model output
    model_weights_path_finetuned = '/data/paintersbynumbers/models/tmp/weights_64_finetuned.11-0.42.h5'

    # dimensions of our images.

    output_shape = 101
    #output_shape=3
    model = vgg16_new.VGG16(model_weights_path,img_size,False)

    top_model = Sequential()
    print(model.output_shape[1:])
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(output_shape, activation='softmax'))
    #top_model.add(BatchNormalization(mode=2))
    #top_model.load_weights(top_model_weights_path)
    model.add(top_model)

    model.load_weights(model_weights_path_finetuned)

    test_pairs = np.load('/data/paintersbynumbers/test_pairs_64.npy')
    Res=[]
    for i in range(test_pairs.shape[0]):
        print('painter pairs -- ' +str(i))
        img1_path=test_pairs[i][0]
        img2_path=test_pairs[i][1]
        print(img1_path+'---'+img2_path)
        img1 = image.load_img(img1_path)
        img2 = image.load_img(img2_path)
        img1 = image.img_to_array(img1)
        img2 = image.img_to_array(img2)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img1 = preprocess_input(img1)
        img2 = preprocess_input(img2)

        x_1 = model.predict(img1)
        x_2 = model.predict(img2)
        #print(x_1)
        #print(x_2)
        dot_product=np.dot(x_1[0],x_2[0])
	#print(x_1[0])
        #print(x_2[0])
        print(dot_product)
        Res.append(dot_product)
    return Res


def computeFeatures_test(filename):

    # path to the model weights file.
    model_weights_path = '/data/paintersbynumbers/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    top_model_weights_path = '/data/paintersbynumbers/models/basic_nn_196.h5'
    data_dir='/data/paintersbynumbers/'
    img_dir = data_dir + filename+'/'
    features_dir = data_dir + 'features/'
    img_size =  196
    output_shape=1584
    model = vgg16_new.VGG16(model_weights_path,img_size)

    top_model = Sequential()
    print(model.output_shape[1:])
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(output_shape, activation='softmax'))
    top_model.add(BatchNormalization(mode=2))
    top_model.load_weights(top_model_weights_path)
    # add the model on top of the convolutional base
    model.add(top_model)
    X=list()
    Y=list()
    paths = os.listdir(img_dir)
    for fname in paths:
        if not fname.startswith('.'):
            img = image.load_img(os.path.join(img_dir, fname),target_size=(img_size,img_size))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            x= model.predict(img)
            y = fname
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

            print(float(len(X)))

    return np.array(X) , np.array(Y)


from sklearn.cross_validation import train_test_split

def train_top_model():
    nb_epoch=50
    train_data = np.load('/data/paintersbynumbers/features/features_train64_100.npy','r')
    train_labels = to_categorical(np.load('/data/paintersbynumbers/features/labels_train64_100.npy','r'))
    top_model_weights_path = '/data/paintersbynumbers/models/basic_nn_64_2.h5'
   #X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    val_data = np.load('/data/paintersbynumbers/features/features_val64_100.npy','r')
    val_labels = to_categorical(np.load('/data/paintersbynumbers/features/labels_val64_100.npy','r'))
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
    print(y_train.shape[1])
    model.add(Dense(y_train.shape[1], activation='softmax'))
    #model.add(BatchNormalization(mode=2))
    #model.load_weights(top_model_weights_path)
    #model.compile( loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.compile( loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    checkpoint = True
    checkpoint = ModelCheckpoint("/data/paintersbynumbers/models/tmp/weights_topmodel.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='max')

    model.fit(X_train,y_train,
              nb_epoch=nb_epoch, batch_size=32,validation_data=(X_val, y_val),callbacks=[checkpoint])
    model.save_weights(top_model_weights_path)
