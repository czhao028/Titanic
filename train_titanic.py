from keras.optimizers import SGD
from keras.layers import BatchNormalization
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import tensorflow as tf
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
    pd_dataframe.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)



activation_function = 'selu'
training_epochs = 4000
learning_rate = 0.343190041904292
num_layers = 4
num_neurons = 594
optimizer = SGD(lr=learning_rate)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

### preprocessing
training_data_raw = pd.read_csv("train.csv")
testing_data_raw = pd.read_csv("test.csv")
data_cleaner(training_data_raw)
data_cleaner(testing_data_raw)

# print(training_data_raw.isna().sum())
# print(testing_data_raw.isna().sum())
#training_input = training_data_raw.dropna(axis=0)

training_output = training_data_raw["Survived"]
training_input = training_data_raw.drop("Survived", axis=1)
#testing_output = testing_data_raw["Survived"]
testing_input = testing_data_raw

# np.asarray(pd_dataframe).astype(np.float32)

model = tf.keras.models.Sequential()
model.add(BatchNormalization())
model.add(tf.keras.Input(shape=(9,)))
for i in range(num_layers):
    model.add(tf.keras.layers.Dense(num_neurons, activation=activation_function))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
x_train, x_holdout, y_train, y_holdout = train_test_split(training_input, training_output, test_size=0.1) #10% holdout as evaluation metric
history = model.fit(x_train, y_train,
    epochs=training_epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_holdout, y_holdout), callbacks=[callback])
print(history.history)
predictions = model.predict(testing_input)
#print(classification_report(testing_output, predictions))

