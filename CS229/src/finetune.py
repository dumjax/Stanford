import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization,Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt
import vgg16_new
from keras import optimizers

def compute(finetuning):
    # path to the model weights files.
    model_weights_path = '/data/paintersbynumbers/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    top_model_weights_path = '/data/paintersbynumbers/models/basic_nn_64_2.h5'
    #fine-tuned model output
    model_weights_path_finetuned = '/data/paintersbynumbers/models/vgg16_nn_64_finetuned.h5'

    # dimensions of our images.
    img_width, img_height = 64, 64

    train_data_dir = '/data/paintersbynumbers/train_64_100'
    validation_data_dir = '/data/paintersbynumbers/val_64_100'
    #train_data_dir = '/data/paintersbynumbers/img_test'
    #validation_data_dir = '/data/paintersbynumbers/img_test'
    nb_train_samples = 2865
    #nb_validation_samples = 8010
    #nb_train_samples = 77
    nb_validation_samples = 306
    nb_epoch = 50

    ## /!\ output dimension
    #output_shape = 1580
    output_shape = 100
    #output_shape=3
    model = vgg16_new.VGG16(model_weights_path,img_width,True)

    top_model = Sequential()
    print(model.output_shape[1:])
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(output_shape, activation='softmax'))
    #top_model.add(BatchNormalization(mode=2))
    top_model.load_weights(top_model_weights_path)
    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if finetuning==True:
        for layer in model.layers[:25]:
            layer.trainable = False
        model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-2, momentum=0.0),metrics=['accuracy'])
    else:
	for layer in model.layers[:32]:
            layer.trainable = False
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
    # fine-tune the modeli
    checkpoint = ModelCheckpoint(filepath='/data/paintersbynumbers/models/tmp/weights_64_finetuned.{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    history=model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            #nb_epoch=nb_epoch,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,callbacks=[checkpoint])

    model.save_weights(model_weights_path_finetuned)
    class_dic = train_generator.class_indices
    np.save('class_dic',class_dic)

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # #plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    #plt.show()


    return history
