# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:09:41 2022

@author: DELL
"""
import numpy as np
import pandas as pd

import seaborn as sns

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

csv_50_df = read_csv_h(csv_50)


raw_dataset = pd.concat([csv_50_df, read_csv_h(csv_100), read_csv_h(csv_200)])
raw_dataset = raw_dataset.append({'depth':0, 'speed':0, 'time':0, 'HResult':0}, ignore_index=True)

dataset = raw_dataset.copy()

h_result = dataset['HResult'].to_numpy()
classes = np.zeros_like(h_result, 'uint8')
classes[np.logical_and(h_result >= 0.015, h_result < 0.05)] = 1
classes[np.logical_and(h_result >= 0.05, h_result < 0.1)] = 2
classes[h_result >= 0.1] = 3
classes[h_result < 0.015] = 0

dataset['class'] = classes
h_result = None
classes = None

dataset.pop('HResult')

### split data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(dataset, diag_kind='auto')
### split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('class')
test_labels = test_features.pop('class')

### Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# reduce the learning rate during training
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=5784*1000,
  decay_rate=1,
  staircase=False)

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)
])

# predictions = model(train_features[:1]).numpy()
# print(predictions)

# # softmax
# print(tf.nn.softmax(predictions).numpy())


model.compile(
    optimizer=tf.optimizers.Adam(lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_features,
    train_labels,
    epochs=2,
    # Suppress logging.
    # verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

model.evaluate(test_features, test_labels)
model.save('d:/tmp/my_model_class')
model1 = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax(),
    ])
test_predictions = model1.predict(csv_50_df[['depth', 'speed', 'time']])