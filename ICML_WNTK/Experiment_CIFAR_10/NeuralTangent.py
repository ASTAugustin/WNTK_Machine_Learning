import numpy as np
import tensorflow as tf
import tqdm, time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

class NeuralTangent:

    def __init__(self,model,x_train,x_test,y_train,y_test,batch_size):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.NTK_Layer = []
        self.K_Layer = []

    @tf.function
    def get_gradient_batch(self,data_batch,W):

        '''
            W: Layer-wise weights of parameters
        '''

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            result = self.model(data_batch)

        gradient_batch = tape.jacobian(result, self.model.trainable_variables)
        del tape

        # Jacobian * W
        gradient_batch = [tf.multiply(g,w) for g,w in zip(gradient_batch,W)]

        return gradient_batch
    
    @tf.function
    def reshape_gradient_batch(self,gradient_batch):
        #reshape
        J = []
        for each in gradient_batch:
            num = 1
            for i in range(1,len(each.shape)):
                num = num * each.shape[i]
            J.append(tf.reshape(each, shape=[each.shape[0],num]))
        J = tf.concat(J, axis=1)
        
        return J

    def get_ntk_full(self,W):

        data = np.append(self.x_train,self.x_test,axis=0)
        X = tf.convert_to_tensor(data,dtype=tf.float32)
        count = 0
        K = []
        results = []

        while X.shape[0] > count * self.batch_size:

            count +=1
            x = X[(count-1)*self.batch_size:min(count*self.batch_size,X.shape[0])]

            gradient_batch = self.get_gradient_batch(data_batch=x,W=W)
            K.append(self.reshape_gradient_batch(gradient_batch).numpy())
            #reshape

        K = np.concatenate(K)   #Jacobian matrix

        self.NTK_Layer = []
        self.K_Layer = []
        
        cursor = 0
        for i in range(len(self.model.trainable_variables)):
            J_train_layer = np.mat(
                                    K[
                                        0:self.x_train.shape[0],
                                        cursor : cursor + np.prod(self.model.trainable_variables[i].shape)
                                    ]
                                )
            J_test_layer = np.mat(
                                    K[
                                        self.x_train.shape[0]:,
                                        cursor : cursor + np.prod(self.model.trainable_variables[i].shape)
                                    ]
                                )
            cursor += np.prod(self.model.trainable_variables[i].shape)

            self.NTK_Layer.append(
                                    J_train_layer * J_train_layer.T
                                )
            self.K_Layer.append(
                                    J_test_layer * J_train_layer.T
                                )
        del K

        NTK = np.sum(self.NTK_Layer, axis=0)    #Theta(X,X)
        K = np.sum(self.K_Layer, axis=0)    #Theta(x,X)

        return NTK, K

    def metrics(self,y_pred,y_true):
        tmp = np.array((y_true * y_pred >0))
        n = np.count_nonzero(tmp)
        return n/y_pred.shape[0]

    def NTK_Ridge_Regression(self,alpha,layer_weights='default'):

        '''
        params:
        alpha: regularization term
        layer_weights: layer-wise weights of network parameters
        '''

        if layer_weights == 'default':
            layer_weights = [1 for i in range(len(self.model.trainable_variables))]

        #get matrices
        NTK, K = self.get_ntk_full(layer_weights)
        
        #regression
        clf = KernelRidge(alpha,kernel='precomputed')
        clf.fit(NTK,self.y_train)
        y_pred = clf.predict(K)
        acc = self.metrics(y_pred,self.y_test)

        return acc

    def update_weight_GD(self, alpha, W_init, learning_rate, iteration, ratio):

        '''
        params:
        alpha: regularization term
        weight: initial weight
        learning_rate: the learning rate of GD on weights
        iteration: number of iteration of GD
        ratio: ratio between train and validation
        '''

        weight = []
        weight.extend(W_init)
        
        print("Update Start!")
        n_variables = []
        for variable in range(len(self.model.trainable_variables)):
            shape = self.model.trainable_variables[variable].get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            n_variables.append(variable_parameters)
        print("Number of Variables:" , n_variables)

        size = self.x_train.shape[0]
        index_train, index_validation = train_test_split(np.arange(size), test_size=ratio)

        for i in tqdm.tqdm(range(iteration), desc='GD Processing'):
            time.sleep(0.01)

            count = 0
            w = []
            for j in range(len(n_variables)):
                w = w + [weight[count] * weight[count]]
                count += 1

            NTK = np.mat(np.sum([np.multiply(ntk_tmp[index_train][:,index_train], w_tmp) for ntk_tmp, w_tmp in zip(self.NTK_Layer, w)], axis=0))
            K = np.mat(np.sum([np.multiply(ntk_tmp[index_validation][:,index_train], w_tmp) for ntk_tmp, w_tmp in zip(self.NTK_Layer, w)], axis=0))
            NTK_inv = np.linalg.inv( NTK + alpha * np.identity(NTK.shape[0]))

            clf = KernelRidge(alpha, kernel='precomputed')
            clf.fit(NTK, self.y_train[index_train])
            y_pred = clf.predict(K)

            term2 = (y_pred - self.y_train[index_validation]).reshape([1,-1])

            for k in range(len(n_variables)):
                w = []
                for j in range(len(n_variables)):
                    if j == k:
                        w = w + [weight[k] * weight[k]]
                    else:
                        w = w + [0]

                NTK_validation = np.mat(np.sum([np.multiply(ntk_tmp[index_train][:,index_train], w_tmp) for ntk_tmp, w_tmp in zip(self.NTK_Layer, w)], axis=0))
                K_validation = np.mat(np.sum([np.multiply(ntk_tmp[index_validation][:,index_train], w_tmp) for ntk_tmp, w_tmp in zip(self.NTK_Layer, w)], axis=0))

                #weights update
                term1 = clf.predict(K_validation).reshape(-1,1) - K * NTK_inv * NTK_validation * NTK_inv * self.y_train[index_train].reshape(-1,1)
                delta = (2) * learning_rate * term2 * term1 
                weight[k] = np.sqrt(max(weight[k] * weight[k] - delta.tolist()[0][0], 0))

            pass
        
        return weight