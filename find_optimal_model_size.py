import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization
from keras.layers import BatchNormalization, LeakyReLU
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def data_cleaner(pd_dataframe):
    pd_dataframe['Age'].fillna(value=pd_dataframe['Age'].median(), inplace=True)
    pd_dataframe['Embarked'].fillna(value=pd_dataframe['Embarked'].mode()[0], inplace=True)
    pd_dataframe['Embarked'] = pd.factorize(pd_dataframe['Embarked'])[0]
    pd_dataframe['Sex'] = pd.factorize(pd_dataframe['Sex'])[0]
    pd_dataframe['Fare'].fillna(value=pd_dataframe['Fare'].median(), inplace=True)
    pd_dataframe['Title'] = pd_dataframe['Name'].str.split(",").str[1].str.split(".").str[0]
    pd_dataframe['Title'] = pd.factorize(pd_dataframe['Title'])[0]
    pd_dataframe['Alone'] = (pd_dataframe['SibSp'] == 0) & (pd_dataframe['Parch'] == 0)

training_data_raw = pd.read_csv("train.csv")
data_cleaner(training_data_raw)
training_input = training_data_raw
training_output = training_input["Survived"]
training_input = training_input.drop(["PassengerId", "Name", "Ticket", "Survived", "Cabin"], axis=1)
training_input = pd.get_dummies(training_input)
training_input = np.asarray(training_input).astype(np.float32)

kfold_model = KFold(n_splits=10, shuffle=True, random_state=123)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

def blackbox_function_find_optimal_parameters(num_neurons, num_layers, activation_function_index,
                                              optimizer_index, learning_rate, normalization, epochs):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'SGD']
    optimizerD = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(lr=learning_rate),
                  'RMSprop': RMSprop(lr=learning_rate), 'Adadelta': Adadelta(lr=learning_rate),
                  'Adagrad': Adagrad(lr=learning_rate), 'Adamax': Adamax(lr=learning_rate),
                  'Nadam': Nadam(lr=learning_rate), 'Ftrl': Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU(alpha=0.1), 'relu']

    num_neurons_int = round(num_neurons)
    num_layers_int = round(num_layers)
    print("Index for activation {0} and optimizer {1}".format(activation_function_index, optimizer_index))
    activation_function_index_model = activationL[round(activation_function_index)]
    optimizerL_name = optimizerL[round(optimizer_index)]
    optimizer_index_model = optimizerD[optimizerL_name]
    loss_fn = BinaryCrossentropy()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(9,)))
    if normalization > 0.5:
        model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(num_neurons_int, activation=activation_function_index_model))
    for num_additional_layer in range(num_layers_int-1):
        model.add(tf.keras.layers.Dense(num_neurons_int, activation=activation_function_index_model))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer_index_model, loss='binary_crossentropy', metrics=['accuracy'])
    x_train, x_test, y_train, y_test = train_test_split(training_input, training_output, test_size=0.1) #K-fold w/ 10
    history = model.fit(x_train, y_train, epochs=round(epochs), callbacks=[callback])
    scores = model.evaluate(x_test, y_test)
    return scores[1] #accuracy on test set

#blackbox_function_find_optimal_parameters(32, 1, 1, 1, 0.01, 0.01, 100)

optimizer = BayesianOptimization(
    f=blackbox_function_find_optimal_parameters,
    pbounds={"num_neurons":(10, 1000), "num_layers":(1,5), "activation_function_index":(0, 9),
                                              "optimizer_index": (0, 8), "learning_rate":(0, 0.5),
             "normalization":(0,1), "epochs":(100, 100000)},
    random_state=123,
)

optimizer.maximize(n_iter=15)

print("Best Parameter Setting : {}".format(optimizer.max["params"]))
print("Best Target Value      : {}".format(optimizer.max["target"]))

