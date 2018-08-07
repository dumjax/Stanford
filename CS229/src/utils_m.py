import sys
import re
import os
import shutil
import commands
import numpy as np


#path_features = '/data/paintersbynumbers/features/'


#features_name = 'features_train256_1.npy'
#labels_name = 'labels_train256_1.npy'

#path_input = '/data/paintersbynumbers/train_256/'

#path_output = '/data/paintersbynumber/train_256_features/'

#X=np.load(path_features +features_name)
#Y=np.load(path_features + labels_name)

def transfer(Y,path_input,path_output):

    for i in range(len(Y)):
        path_folder_output = path_output+str(Y[i])
        if not os.path.exists(path_folder_output):
            os.mkdir(path_folder_output)
            print('input --- ' + path_input+str(Y[i]))
            print('output --- ' + path_folder_output)
            copy_to(path_input+str(Y[i]),path_folder_output)
        else:
            print(path_folder_output + '--- already exists')




def copy_to(path_input, to_dir):
    paths = os.listdir(path_input)
    for path in paths:
        print(path)
        fname = os.path.basename(path)
        print(fname)
        if not fname.startswith('.'):
            shutil.copy(os.path.abspath(os.path.join(path_input, fname)), os.path.join(to_dir, fname))
