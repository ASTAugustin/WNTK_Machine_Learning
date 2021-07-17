import tensorflow as tf
import matplotlib.pyplot as plt
import time

def multiplier(x, mul=0.01):
    return x * mul

class CNN_5(tf.keras.Sequential):
    def __init__(self, layers=None, name=None, input_shape=None):
        super().__init__(layers=layers, name=name)
        
        self.add(tf.keras.layers.Conv2D(64, 3, input_shape=input_shape, activation=tf.nn.tanh))
        self.add(tf.keras.layers.Conv2D(64, 3, activation=tf.nn.tanh))
        self.add(tf.keras.layers.Conv2D(64, 3, activation=tf.nn.tanh))
        self.add(tf.keras.layers.Conv2D(64, 3, activation=tf.nn.tanh))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1,activation=multiplier))

    #@tf.function
    def loss(self, y_pred, y_true):
        L = tf.keras.losses.MeanSquaredError()(y_pred,y_true)
        return L

# @tf.function
def train_step(x_batch, y_batch, model:CNN_5, OP:tf.keras.optimizers.Optimizer, metrics:list, layer_lr:list):
    with tf.GradientTape() as tape:
        y_batch_pred = model(x_batch)
        loss = model.loss(y_batch_pred,y_batch)
    grad = tape.gradient(loss, model.trainable_variables)
    grad = [tf.multiply(g,w) for g,w in zip(grad,layer_lr)]
    OP.apply_gradients(zip(grad, model.trainable_variables))

    metrics[0](loss)
    metrics[1]((y_batch+1)/2,y_batch_pred)

# @tf.function
def test_step(x_batch, y_batch, model:CNN_5, metrics:list):
    y_batch_pred = model(x_batch)
    loss = model.loss(y_batch_pred,y_batch)
    
    metrics[0](loss)
    metrics[1]((y_batch+1)/2,y_batch_pred)

def train(train_data, test_data, epochs, model:CNN_5, OP:tf.keras.optimizers.Optimizer, layer_lr:list, verbose=1):

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy',threshold=0)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy',threshold=0)

    train_l = []
    train_a = []
    test_l = []
    test_a = []
    
    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x_batch, y_batch in train_data:
            train_step(x_batch, y_batch, model, OP, metrics=[train_loss,train_accuracy], layer_lr=layer_lr)

        for x_batch, y_batch in test_data:
            test_step(x_batch, y_batch, model, metrics=[test_loss,test_accuracy])

        if verbose==1:
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print (template.format(epoch+1,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    test_loss.result(),
                                    test_accuracy.result()*100))
        train_l.append(train_loss.result())
        train_a.append(train_accuracy.result())
        test_l.append(test_loss.result())
        test_a.append(test_accuracy.result())

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epochs,
            train_loss.result(),
            train_accuracy.result()*100,
            test_loss.result(),
            test_accuracy.result()*100))

    return train_l, train_a, test_l, test_a

def train_step_adam(x_batch, y_batch, model:CNN_5, OPs, metrics:list):
    with tf.GradientTape(persistent=True) as tape:
        y_batch_pred = model(x_batch)
        loss = model.loss(y_batch_pred,y_batch)
    grads = []
    for i in range(len(OPs)):
        grad = tape.gradient(loss, [model.trainable_variables[i]])
        grads.append(grad)
    for i in range(len(OPs)):
        OPs[i].apply_gradients(zip(grads[i], [model.trainable_variables[i]]))
    del tape
    metrics[0](loss)
    metrics[1]((y_batch+1)/2,y_batch_pred)

def train_adam(train_data, test_data, epochs, model:CNN_5, OPs, verbose=1):

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy',threshold=0)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy',threshold=0)

    train_l = []
    train_a = []
    test_l = []
    test_a = []
    
    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x_batch, y_batch in train_data:
            train_step_adam(x_batch, y_batch, model, OPs, metrics=[train_loss,train_accuracy])

        for x_batch, y_batch in test_data:
            test_step(x_batch, y_batch, model, metrics=[test_loss,test_accuracy])

        if verbose==1:
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print (template.format(epoch+1,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    test_loss.result(),
                                    test_accuracy.result()*100))
        train_l.append(train_loss.result())
        train_a.append(train_accuracy.result())
        test_l.append(test_loss.result())
        test_a.append(test_accuracy.result())

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epochs,
            train_loss.result(),
            train_accuracy.result()*100,
            test_loss.result(),
            test_accuracy.result()*100))

    return train_l, train_a, test_l, test_a
