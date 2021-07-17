#hardware setting (not necessary)
#region
import os
#gpu usage setting

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing


import tensorflow as tf
#gpu memory setting

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#endregion

import numpy as np
import sys
import math
from utils import CIFAR10_load, Logger

def CIFAR10_sub(samples,class_1,class_2):

    #class:|0:airplane|1:automobile|2:bird|3:cat|4:deer|5:dog|6:frog|7:horse|8:ship|9:truck|
    
    cifar10_path = "data_cifar10"   # Cifar-10 dataset path 
    x_train, y_train, x_test, y_test = CIFAR10_load(cifar10_path)

    #training set subsample, only using class_1 and class_2
    sub = samples
    index_0 = np.argwhere(y_train == class_1)
    index_1 = np.argwhere(y_train == class_2)
    index = np.append(index_0[0:int(samples/2)],index_1[0:int(samples/2)],axis=0)
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    #test set subsample, only using class_1 and class_2
    index_0 = np.argwhere(y_test == class_1)
    index_1 = np.argwhere(y_test == class_2)
    index = np.append(index_0,index_1,axis=0)
    np.random.shuffle(index)
    x_test = x_test[index]
    y_test = y_test[index]

    #reshape
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[2],x_train.shape[3],x_train.shape[4])
    y_train = 2 * (y_train.reshape(y_train.shape[0])-(class_1+class_2)/2) / abs(class_2-class_1)
    y_train = np.array(y_train,dtype='int8')

    x_test = x_test.reshape(x_test.shape[0],x_test.shape[2],x_test.shape[3],x_test.shape[4])
    y_test = 2 * (y_test.reshape(y_test.shape[0])-(class_1+class_2)/2) / abs(class_2-class_1)
    y_test = np.array(y_test,dtype='int8')

    return x_train,y_train,x_test,y_test

sys.stdout = Logger(sys.stdout)

def regression_accuracy(y_pred,y_true):
    tmp = np.array((y_true * y_pred >0))
    n = np.count_nonzero(tmp)
    return n / y_pred.shape[0]


class_Cifar=[(0,1),(2,3),(3,4),(4,5)]
for j in range(4):

    x_train, y_train, x_test, y_test = CIFAR10_sub(10000,class_Cifar[j][0],class_Cifar[j][1])
    x_train, y_train, x_test, y_test = x_train*1.0, y_train*1.0, x_test*1.0, y_test*1.0

    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape[0],-1))

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train) 
    x_test = scaler.transform(x_test) 

    clf_gaussian = KernelRidge(1,kernel='rbf')
    clf_gaussian.fit(x_train,y_train)
    y_pred_gaussian = clf_gaussian.predict(x_test)

    clf_poly = KernelRidge(1,kernel='poly')
    clf_poly.fit(x_train,y_train)
    y_pred_poly = clf_poly.predict(x_test)

    clf_sigmoid = KernelRidge(1,kernel='sigmoid')
    clf_sigmoid.fit(x_train,y_train)
    y_pred_sigmoid = clf_sigmoid.predict(x_test)

    clf_laplacian = KernelRidge(1,kernel='laplacian')
    clf_laplacian.fit(x_train,y_train)
    y_pred_laplacian = clf_laplacian.predict(x_test)

    Gaussian_metrics = regression_accuracy(y_pred_gaussian,y_test)
    Laplacian_metrics = regression_accuracy(y_pred_laplacian,y_test)
    Sigmoid_metrics = regression_accuracy(y_pred_sigmoid,y_test)
    Poly_metrics = regression_accuracy(y_pred_poly,y_test)

    print('Laplacian:', Laplacian_metrics)
    print('Gaussian:', Gaussian_metrics)
    print('Sigmoid:', Sigmoid_metrics)
    print('Polynomial:', Poly_metrics)