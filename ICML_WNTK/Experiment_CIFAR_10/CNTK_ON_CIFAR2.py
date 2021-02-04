#hardware setting (not necessary)
#region
import os
#gpu usage setting

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

#import
#region
import numpy as np
import sys
import math
from utils import CIFAR10_load, Logger
from NeuralTangent import NeuralTangent
#endregion

#Network Custom Functions
#region

#metrics
def accuracy(y_pred,y_true):
    length = tf.cast(tf.shape(y_true)[0], tf.int64)
    tmp = tf.reshape((y_true * y_pred >0),(-1,))
    n = tf.math.count_nonzero(tmp)
    return tf.divide(n, length)

#activation
def multiplier(x, mul=0.01):
    return x * mul

#callbacks
class ThresholdStopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdStopping, self).__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["val_accuracy"]
        if accuracy >= self.threshold:
            self.model.stop_training = True

#endregion

#params
#region

#net params
L = 3       # hidden layers (network depth = L + 2)
M = 64      # network width = M

#NTK params
NTK_params = {
            'W_init': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1], 
            'lr': 0.01, 
            'iteration': 50, 
            'val_ratio': 0.2, 
            'NN_stop_threshold': 0.6
            }

#network pre-training params
samples = 100
batch_size = pow( 2, 1 + round(math.log(samples,10)) )
epochs = 20

#endregion

#log
sys.stdout = Logger(sys.stdout)

#data loading
#region
cifar10_path = 'Experiment_CIFAR_10\cifar-10-batches-py'    #dataset path
x_train, y_train, x_test, y_test = CIFAR10_load(cifar10_path)
#class:|0:airplane|1:automobile|2:bird|3:cat|4:deer|5:dog|6:frog|7:horse|8:ship|9:truck|
class_1 = 0
class_2 = 1

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

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#endregion

NTK_pure_acc = []
WNTK_pure_acc = []
NTK_init_acc = []
WNTK_init_acc = []

for ite in range(10):

    #network model
    #region    
    Net = tf.keras.Sequential()
    Net.add(tf.keras.layers.Conv2D(M, 3, input_shape=x_train.shape[1:], activation=tf.nn.tanh))
    for i in range(L):
        Net.add(tf.keras.layers.Conv2D(M, 3, activation=tf.nn.tanh))
    Net.add(tf.keras.layers.Flatten())
    Net.add(tf.keras.layers.Dense(1, activation=multiplier))

    Net.compile(
                optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[accuracy]
                )
    print(Net.summary())
    #endregion

    #WNTK and NTK
    #region
    #create neural tangent kernel
    NT = NeuralTangent(Net,x_train,x_test,y_train,y_test,batch_size=32)

    #NTK Regression Performance
    NTK_pure = NT.NTK_Ridge_Regression(alpha=1)
    print('Initial CNTK accuracy: ', NTK_pure)

    #Update_weight_GD
    print('Layer weights start at', NTK_params['W_init'])
    W_new = NT.update_weight_GD(1,NTK_params['W_init'], NTK_params['lr'], NTK_params['iteration'], NTK_params['val_ratio'])
    W_new = np.array(W_new).astype(np.float32).tolist()
    print('Final layer weights after GD:',W_new)

    #NTK Regression Performance
    WNTK_pure = NT.NTK_Ridge_Regression(alpha=1,layer_weights=W_new)
    print('CNTK accuracy with layer-weights After GD: ', WNTK_pure)
    #endregion

    #network pre-training trick
    #region
    history = Net.fit(
                        x_train,y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=NTK_params['val_ratio'],
                        #validation_data=(x_test,y_test),
                        verbose=2,
                        callbacks = 
                        [
                            ThresholdStopping(threshold = NTK_params['NN_stop_threshold']),
                        ]
                    )
    print(history.history.keys())
    #endregion

    #WNTK after network pre-training trick
    #region
    #create neural tangent kernel
    NT = NeuralTangent(Net,x_train,x_test,y_train,y_test,batch_size=32)

    #NTK Regression Performance
    NTK_init = NT.NTK_Ridge_Regression(alpha=1)
    print('After training, CNTK accuracy: ', NTK_init)

    #Update_weight_GD
    print('Layer weights start at', NTK_params['W_init'])
    W_new = NT.update_weight_GD(1, NTK_params['W_init'], NTK_params['lr'], NTK_params['iteration'], NTK_params['val_ratio'])
    W_new = np.array(W_new).astype(np.float32).tolist()
    print('Final layer weights after GD:',W_new)

    #NTK Regression Performance
    WNTK_init = NT.NTK_Ridge_Regression(alpha=1,layer_weights=W_new)
    print('After training, CNTK accuracy with layer-weights: ', WNTK_init)
    #endregion

    NTK_pure_acc.append(NTK_pure)
    WNTK_pure_acc.append(WNTK_pure)
    NTK_init_acc.append(NTK_init)
    WNTK_init_acc.append(WNTK_init)

#Summary
#region
print('Network architecture:')
print(Net.summary())
print('Class:',class_1,class_2)
print('Training samples:', x_train.shape[0])
print('Test samples:', x_test.shape[0])
print('NTK Parameters:', NTK_params)

print("NTK_pure_acc:" + str(NTK_pure_acc))
print("WNTK_pure_acc:" + str(WNTK_pure_acc))
print("NTK_init_acc:" + str(NTK_init_acc))
print("WNTK_init_acc:" + str(WNTK_init_acc))

print("NTK_pure_acc mean & std:", np.mean(NTK_pure_acc), np.std(NTK_pure_acc))
print("WNTK_pure_acc mean & std:", np.mean(WNTK_pure_acc), np.std(WNTK_pure_acc))
print("NTK_init_acc mean & std:", np.mean(NTK_init_acc), np.std(NTK_init_acc))
print("WNTK_init_acc mean & std:", np.mean(WNTK_init_acc), np.std(WNTK_init_acc))
#endregion
