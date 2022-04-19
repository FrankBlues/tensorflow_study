# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:09:41 2022

@author: DELL
"""
import os

import numpy as np
import pandas as pd


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def read_csv_h(csv):
    df = pd.read_csv(csv)
    return df[['depth', 'speed', 'time', 'HResult']]

csv_200 = 'd:/tmp/风险200.csv'
csv_50 = 'd:/tmp/风险50.csv'
csv_100 = 'd:/tmp/风险100.csv'
# column_names = ['OBJECTID', 'depth', 'speed', 'time', 'year', 'HResult']


raw_dataset = pd.concat([read_csv_h(csv_50), read_csv_h(csv_100), read_csv_h(csv_200)])
# raw_dataset = raw_dataset.append({'depth':0, 'speed':0, 'time':0, 'HResult':0}, ignore_index=True)

dataset = raw_dataset.copy()

# drop some value
dataset = dataset.drop(dataset[dataset['time'] == 0].index)

### split data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# train_dataset = dataset
# test_dataset = raw_dataset_50[['depth', 'speed', 'time', 'HResult']]

# sns.pairplot(dataset, diag_kind='auto')
### split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('HResult')
test_labels = test_features.pop('HResult')

### Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# reduce the learning rate during training
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=3787*3000,
  decay_rate=1,
  staircase=False)

batch_size = 40
# Create a callback that saves the model's weights every 5 epochs
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=100*batch_size)

# model
dnn_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
# print(linear_model.summary())

dnn_model.compile(
    optimizer=tf.optimizers.Adam(lr_schedule),
    loss='mean_absolute_error')

history = dnn_model.fit(
    train_features,
    train_labels,
    epochs=3000,
    callbacks=[cp_callback],
    # Suppress logging.
    # verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

dnn_model.evaluate(test_features, test_labels)
dnn_model.save('d:/tmp/my_model_drop0')
test_predictions = dnn_model.predict(test_features).flatten()