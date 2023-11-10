import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.losses import BinaryCrossentropy
from torch.nn import LeakyReLU
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

training_data_raw = pd.read_csv("train.csv")
training_input = training_data_raw.drop(["PassengerId", "Name", "Ticket", "Survived"], axis=1)
training_input = pd.get_dummies(training_input)
training_output = training_data_raw["Survived"]

kfold_model = KFold(n_splits=10, shuffle=True, random_state=123)

def blackbox_function_find_optimal_parameters(num_neurons, num_layers, activation_function_index,
                                              optimizer_index, learning_rate, normalization):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'SGD']
    optimizerD = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(lr=learning_rate),
                  'RMSprop': RMSprop(lr=learning_rate), 'Adadelta': Adadelta(lr=learning_rate),
                  'Adagrad': Adagrad(lr=learning_rate), 'Adamax': Adamax(lr=learning_rate),
                  'Nadam': Nadam(lr=learning_rate), 'Ftrl': Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']

    num_neurons_int = round(num_neurons)
    num_layers_int = round(num_layers)
    activation_function_index_model = activationL[round(activation_function_index)]
    optimizer_index_model = optimizerD[round(optimizer_index)]
    loss_fn = BinaryCrossentropy(from_logits=True)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    if normalization > 0.5:
        model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(num_neurons_int, activation=activation_function_index_model))
    for num_additional_layer in range(num_layers_int-1):
        model.add(tf.keras.layers.Dense(num_neurons_int, activation=activation_function_index_model))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer_index_model, loss=loss_fn, metrics=['accuracy'])
    x_train, x_test, y_train, y_test = train_test_split(training_input, training_output, test_size=0.1) #K-fold w/ 10
    history = model.fit(x_train, y_train, epochs=100)
    scores = model.evaluate(x_test, y_test)

blackbox_function_find_optimal_parameters(32, 1, 1, 0.01, 0.01)



