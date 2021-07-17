#hardware setting (not necessary)
#region
import os
import time
import tensorflow as tf

#gpu usage setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from model import CNN_5, train, train_adam
from NeuralTangent import NeuralTangent
import matplotlib.pyplot as plt

def CIFAR10_sub(samples,class_1,class_2):

    #class:|0:airplane|1:automobile|2:bird|3:cat|4:deer|5:dog|6:frog|7:horse|8:ship|9:truck|
    
    cifar10_path = 'data_cifar10'    #dataset path 
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

NTK_params = {
            'W_init': [0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1], 
            'lr': 0.01,
            'iteration': 100, 
            'val_ratio': 0.2, 
            'NN_stop_threshold': 0.6
            }

# class_Cifar=[(0,1),(2,3),(3,4),(4,5)]
class_Cifar = [(2,3)]

for j in range(len(class_Cifar)):

    x_train, y_train, x_test, y_test = CIFAR10_sub(10000,class_Cifar[j][0],class_Cifar[j][1])
    x_train, y_train, x_test, y_test = x_train*1.0, y_train*1.0, x_test*1.0, y_test*1.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(100).batch(32)
    test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)
    x_NTK, y_NTK = x_train[0:1000], y_train[0:1000]

    SGD_epoch = 1000
    RMS_epoch = 250
    Adam_epoch = 250

    NTK_acc = []
    WNTK_acc = []
    NN_SGD_acc = []
    NN_LayerLR_acc = []
    NN_Adam_acc = []
    NN_LayerLR_Adam_acc = []
    NN_RMS_acc = []
    NN_LayerLR_RMS_acc = []

    for i in range(1):

        print("Iteration", i)
        Net = CNN_5(input_shape=x_train.shape[1:])
        Net.save_weights('./CNN_Init')


        NT = NeuralTangent(Net,x_NTK,[],y_NTK,[],batch_size=32)

        #NTK
        #region

        print("calculating CNTK Regression")
        NTK_pure = NT.NTK_Ridge_Regression(alpha=1)
        print('Initial CNTK accuracy: ', NTK_pure*100)
        NTK_acc.append(NTK_pure*100)

        #endregion
    
        #WNTK
        #region

        print('Layer weights start at', NTK_params['W_init'])
        W_new = NT.update_weight_GD(1, NTK_params['W_init'], NTK_params['lr'], NTK_params['iteration'], NTK_params['val_ratio'])
        W_new = np.array(W_new).astype(np.float32).tolist()
        print('Final layer weights after GD:',W_new)
        WNTK_pure = NT.NTK_Ridge_Regression(alpha=1, layer_weights=W_new)
        print('CNTK accuracy with layer-weights After GD: ', WNTK_pure*100)
        WNTK_acc.append(WNTK_pure*100)

        # endregion

        default_lr = np.ones(10)
        new_lr = W_new/np.mean(np.array(W_new))
        print("Adjusted Learning Rate:", new_lr)


        #RMS_LayerLR
        #region
        op = tf.keras.optimizers.RMSprop(0.001)
        Net.load_weights('./CNN_Init')
        train_l_RMS, train_a_RMS, test_l_RMS, test_a_RMS = train(train_data, test_data, RMS_epoch, Net, op, layer_lr=default_lr, verbose=1)
        NN_RMS_acc.append(test_a_RMS[-1])
        #endregion

        #RMS
        #region
        op_list = []
        for j in range(len(new_lr)):
            op_list.append(tf.keras.optimizers.RMSprop(learning_rate=0.001 * new_lr[j]))
        Net.load_weights('./CNN_Init')
        train_l_lr_RMS, train_a_lr_RMS, test_l_lr_RMS, test_a_lr_RMS = train_adam(train_data, test_data, RMS_epoch, Net, op_list, verbose=1)
        NN_LayerLR_RMS_acc.append(test_a_lr_RMS[-1])
        #endregion

        #Adam_LayerLR
        #region
        op = tf.keras.optimizers.Adam(0.001)
        Net.load_weights('./CNN_Init')
        train_l_Adam, train_a_Adam, test_l_Adam, test_a_Adam = train(train_data, test_data, Adam_epoch, Net, op, layer_lr=default_lr, verbose=1)
        NN_Adam_acc.append(test_a_Adam[-1])
        #endregion

        #Adam
        #region
        op_list = []
        for j in range(len(new_lr)):
            op_list.append(tf.keras.optimizers.Adam(learning_rate=0.001 * new_lr[j]))
        Net.load_weights('./CNN_Init')
        train_l_lr_Adam, train_a_lr_Adam, test_l_lr_Adam, test_a_lr_Adam = train_adam(train_data, test_data, Adam_epoch, Net, op_list, verbose=1)
        NN_LayerLR_Adam_acc.append(test_a_lr_Adam[-1])
        #endregion


        #SGD
        #region
        op = tf.keras.optimizers.SGD(0.001)
        Net.load_weights('./CNN_Init')
        train_l, train_a, test_l, test_a = train(train_data, test_data, SGD_epoch, Net, op, layer_lr=default_lr, verbose=1)
        NN_SGD_acc.append(test_a[-1])
        #endregion

        #SGD_LayerLR
        #region
        op = tf.keras.optimizers.SGD(0.001)
        Net.load_weights('./CNN_Init')
        train_l_lr, train_a_lr, test_l_lr, test_a_lr = train(train_data, test_data, SGD_epoch, Net, op, layer_lr=new_lr, verbose=1)
        NN_LayerLR_acc.append(test_a_lr[-1])
        #endregion

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(SGD_epoch), train_l, color="r", linestyle="-", linewidth=1, label = "SGD")
        ax.plot(range(SGD_epoch), train_l_lr, color="b", linestyle="-", linewidth=1, label = "SGD_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("train_loss")
        plt.title('SGD Train Loss')
        ax.legend(loc='upper right')
        plt.savefig('./Figures_ML/SGD_Train_Loss_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/SGD_Train_Loss_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(Adam_epoch), train_l_Adam, color="y", linestyle="-", linewidth=1, label = "Adam")
        ax.plot(range(Adam_epoch), train_l_lr_Adam, color="g", linestyle="-", linewidth=1, label = "Adam_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("train_loss")
        plt.title('Adam Train Loss')
        ax.legend(loc='upper right')
        plt.savefig('./Figures_ML/Adam_Train_Loss_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/Adam_Train_Loss_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(RMS_epoch), train_l_RMS, color="y", linestyle="-", linewidth=1, label = "RMSprop")
        ax.plot(range(RMS_epoch), train_l_lr_RMS, color="g", linestyle="-", linewidth=1, label = "RMSprop_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("train_loss")
        plt.title('RMSprop Train Loss')
        ax.legend(loc='upper right')
        plt.savefig('./Figures_ML/RMSprop_Train_Loss_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/RMSprop_Train_Loss_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(SGD_epoch), train_a, color="r", linestyle="-", linewidth=1, label = "SGD")
        ax.plot(range(SGD_epoch), train_a_lr, color="b", linestyle="-", linewidth=1, label = "SGD_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("train_accuracy")
        plt.title('SGD Train Accuracy')
        ax.legend(loc='lower right')
        plt.savefig('./Figures_ML/SGD_Train_Accuracy_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/SGD_Train_Accuracy_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(Adam_epoch), train_a_Adam, color="y", linestyle="-", linewidth=1, label = "Adam")
        ax.plot(range(Adam_epoch), train_a_lr_Adam, color="g", linestyle="-", linewidth=1, label = "Adam_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("train_accuracy")
        plt.title('Adam Train Accuracy')
        ax.legend(loc='lower right')
        plt.savefig('./Figures_ML/Adam_Train_Accuracy_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/Adam_Train_Accuracy_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(RMS_epoch), train_a_RMS, color="y", linestyle="-", linewidth=1, label = "RMSprop")
        ax.plot(range(RMS_epoch), train_a_lr_RMS, color="g", linestyle="-", linewidth=1, label = "RMSprop_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("train_accuracy")
        plt.title('RMSprop Train Accuracy')
        ax.legend(loc='lower right')
        plt.savefig('./Figures_ML/RMSprop_Train_Accuracy_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/RMSprop_Train_Accuracy_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(SGD_epoch), test_l, color="r", linestyle="-", linewidth=1, label = "SGD")
        ax.plot(range(SGD_epoch), test_l_lr, color="b", linestyle="-", linewidth=1, label = "SGD_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("test_loss")
        plt.title('SGD Test Loss')
        ax.legend(loc='upper right')
        plt.savefig('./Figures_ML/SGD_Test_Loss_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/SGD_Test_Loss_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(Adam_epoch), test_l_Adam, color="y", linestyle="-", linewidth=1, label = "Adam")
        ax.plot(range(Adam_epoch), test_l_lr_Adam, color="g", linestyle="-", linewidth=1, label = "Adam_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("test_loss")
        plt.title('Adam Test Loss')
        ax.legend(loc='upper right')
        plt.savefig('./Figures_ML/Adam_Test_Loss_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/Adam_Test_Loss_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(RMS_epoch), test_l_RMS, color="y", linestyle="-", linewidth=1, label = "RMSprop")
        ax.plot(range(RMS_epoch), test_l_lr_RMS, color="g", linestyle="-", linewidth=1, label = "RMSprop_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("test_loss")
        plt.title('RMSprop Test Loss')
        ax.legend(loc='upper right')
        plt.savefig('./Figures_ML/RMSprop_Test_Loss_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/RMSprop_Test_Loss_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(SGD_epoch), test_a, color="r", linestyle="-", linewidth=1, label = "SGD")
        ax.plot(range(SGD_epoch), test_a_lr, color="b", linestyle="-", linewidth=1, label = "SGD_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("test_accuracy")
        plt.title('SGD Test Accuracy')
        ax.legend(loc='lower right')
        plt.savefig('./Figures_ML/SGD_Test_Accuracy_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/SGD_Test_Accuracy_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(Adam_epoch), test_a_Adam, color="y", linestyle="-", linewidth=1, label = "Adagrad")
        ax.plot(range(Adam_epoch), test_a_lr_Adam, color="g", linestyle="-", linewidth=1, label = "Adagrad_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("test_accuracy")
        plt.title('Adam Test Accuracy')
        ax.legend(loc='lower right')
        plt.savefig('./Figures_ML/Adam_Test_Accuracy_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/Adam_Test_Accuracy_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()

        plt.cla()
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(range(RMS_epoch), test_a_RMS, color="y", linestyle="-", linewidth=1, label = "RMSprop")
        ax.plot(range(RMS_epoch), test_a_lr_RMS, color="g", linestyle="-", linewidth=1, label = "RMSprop_adjusted")
        plt.xlabel("epoch")
        plt.ylabel("test_accuracy")
        plt.title('RMSprop Test Accuracy')
        ax.legend(loc='lower right')
        plt.savefig('./Figures_ML/RMSprop_Test_Accuracy_{}.eps'.format(time.strftime('%Y-%m-%d-%H-%M')),dpi=1)
        plt.savefig('./Figures_PNG/RMSprop_Test_Accuracy_{}.png'.format(time.strftime('%Y-%m-%d-%H-%M')))
        plt.show()


    print('NTK:',np.mean(NTK_acc),np.std(NTK_acc))
    print('WNTK:',np.mean(WNTK_acc),np.std(WNTK_acc))
    print('SGD:', np.mean(NN_SGD_acc), np.std(NN_SGD_acc))
    print('SGD LayerLR:', np.mean(NN_LayerLR_acc), np.std(NN_LayerLR_acc))
    print('Adam:', np.mean(NN_Adam_acc), np.std(NN_Adam_acc))
    print('Adam LayerLR:', np.mean(NN_LayerLR_Adam_acc), np.std(NN_LayerLR_Adam_acc))
    print('RMSprop:', np.mean(NN_RMS_acc), np.std(NN_RMS_acc))
    print('RMSprop LayerLR:', np.mean(NN_LayerLR_RMS_acc), np.std(NN_LayerLR_RMS_acc))
