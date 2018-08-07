import sys, os
#sys.path.insert(0, "~/cs229_project/deep-learning-models/")

import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model.tests.test_randomized_l1
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from vgg16 import VGG16
#from imagenet_utils import preprocess_input, decode_predictions
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

dataset_dir = '/data/paintersbynumbers/features/'
test_ratio=0.2
X = np.load(dataset_dir + 'features_train64_100.npy')
#X = np.load(dataset_dir + 'features_train256_100.npy')
y = np.load(dataset_dir + 'labels_train64_finetuned.npy')
X_test=np.load(dataset_dir + 'features_test64_100.npy')
y_test=np.load(dataset_dir + 'labels_test64_finetuned.npy')
rseed=42
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rseed)

print("Total images: ", X.shape[0])

## learning
#non-linear svm implemenatation  accuracy = 
X = X.ravel().reshape(X.shape[0],-1)
X_test=X_test.ravel().reshape(X_test.shape[0],-1)
print('Data import complete.')
print('X:',X.shape)
print('y:',y.shape)
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)
y_test.shape
#X_test= X_test.ravel().reshape(X_test.shape[0],-1)
#y_train_oneDim = np.dot(y_train,[1,2,3,4,5,6,7,8,9])
#y_test_oneDim = np.dot(y_test,[1,2,3,4,5,6,7,8,9])
kernel_clf = svm.SVC(decision_function_shape='ovr')
kernel_clf.fit(X, y)
kernel_train = kernel_clf.score(X,y)
kernel_test = kernel_clf.score(X_test,y_test)
print("Non-linear SVM train error",kernel_train)
print("Non-linear SVM test error", kernel_test)

#linear svm  accuracy =
lin_clf = svm.LinearSVC(loss = 'hinge')
lin_clf.fit(X,y)
lin_test=lin_clf.score(X_test,y_test)
lin_train=lin_clf.score(X,y)
print("Linear SVM train error",lin_train)
print("Linear SVM test error",lin_test)

from sklearn.externals import joblib
#joblib.dump(lin_clf, 'lin_clf_svm_weights_.pkl') 
#lin_clf = joblib.load('lin_clf_svm_weights.pkl')
#svm_predictions = lin_clf.predict(X_test)
#np.save('/data/paintersbynumbers/features/svm_predictions_.npy', svm_predictions)
#random forest implementation  accuracy =
rf_clf = RandomForestClassifier(n_estimators=10)
rf_clf.fit(X,y)
rf_test=rf_clf.score(X_test,y_test)
rf_train=rf_clf.score(X,y)
print("Random Forest train error", rf_train)
print("Random Forest train error",rf_test)


#SDG implementation with hinge loss (SVM)
#from sklearn.linear_model import SGDClassifier
#sgd_clf = SGDClassifier(loss="hinge",penalty="l2")
#sgd_clf.fit(X,y)
#sgd_train=sgd_clf.score(X,y)
#sgd_test=sgd_clf.score(X_test,y_test)
#print('SGD train error',sgd_train)
#print('SGD test error',sgd_test)

