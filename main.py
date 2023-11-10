import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))