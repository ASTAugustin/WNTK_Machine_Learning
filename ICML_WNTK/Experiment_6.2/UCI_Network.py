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
from sklearn.model_selection import StratifiedKFold
from utils import Logger
#endregion

#callbacks
class FullTraining(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(FullTraining, self).__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["accuracy"]
        if accuracy < self.threshold:
            self.model.stop_training = False

#activation
def multiplier(x, mul=0.01):
    return x * mul

#dataset path
datadir = 'Experiment_UCI//data'

#log
sys.stdout = Logger(sys.stdout)

Dataset = []
Performance = []

#k-fold cross validation for every dataset
for idx, dataset in enumerate(sorted(os.listdir(datadir))):

    dic = dict()
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test
    
    print(idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

    # load data
    f = open("Experiment_UCI//data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    Y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))

    folder = StratifiedKFold(n_splits=4)

    #Cross Validation
    #region
    NN_trained_acc = []

    for train, val in folder.split(X,Y):
        #MLP model
        #region
        Net = tf.keras.Sequential()
        Net.add(tf.keras.layers.Dense(1024, input_shape=(X.shape[1],), activation=tf.nn.relu))
        for i in range(3):
            Net.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
        Net.add(tf.keras.layers.Dense(1, activation=multiplier))

        Net.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy']
                    )
        print(Net.summary())
        #endregion
        
        #traning
        #region
        x_train = X[train]
        y_train = Y[train]
        x_val = X[val]
        y_val = Y[val]
        history = Net.fit(
                            x_train,y_train,
                            epochs=1000,
                            batch_size=8,
                            validation_data=(x_val,y_val),
                            verbose=2,
                            callbacks=
                            [
                                tf.keras.callbacks.EarlyStopping(
                                                                monitor='val_accuracy', 
                                                                min_delta=0.0001, 
                                                                patience=20, 
                                                                verbose=0, 
                                                                mode='max', 
                                                                baseline=None, 
                                                                restore_best_weights=False
                                                                )
                                ,FullTraining(0.9990)
                            ]
                        )
        NN_trained_acc.append(history.history['val_accuracy'][-1])
        #endregion

    print(idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)
    print('Accuracy mean & std:', np.mean(NN_trained_acc), np.std(NN_trained_acc))

    Dataset.append([idx, dataset])
    Performance.append([np.mean(NN_trained_acc), np.std(NN_trained_acc)])
    #endregion

#summary
print('Dataset:', Dataset)
print('Performance:', Performance)