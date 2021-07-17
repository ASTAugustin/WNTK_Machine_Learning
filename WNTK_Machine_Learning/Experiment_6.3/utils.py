import struct
import sys
import os
import time
import pickle
import numpy as np

#log output
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "CIFAR_log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#For CIFAR10 loading
#|0:airplane|1:automobile|2:bird|3:cat|4:deer|5:dog|6:frog|7:horse|8:ship|9:truck|
#region
def CIFAR10_load(path):
    #for training set
    x_train = np.empty((50000,32,32,3), dtype='uint8')
    y_train = np.empty(50000, dtype='uint8')
    for i in range(1,6):
        with open(os.path.join(path, 'data_batch_' + str(i)),'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_train[(i-1)*10000:i*10000,:,:,:] = np.array(dict[b'data']).reshape(10000,3,32,32).transpose(0,2,3,1)
            y_train[(i-1)*10000:i*10000] = np.array(dict[b'labels'])
            #x_train[(i-1)*10000:i*10000,:,:,:], y_train[(i-1)*10000:i*10000] = dict

    #for test set
    with open(os.path.join(path, 'test_batch'),'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x_test = np.array(dict[b'data']).reshape(10000,3,32,32).transpose(0,2,3,1)
        y_test = np.array(dict[b'labels'])

    return x_train, y_train, x_test, y_test
#endregion
