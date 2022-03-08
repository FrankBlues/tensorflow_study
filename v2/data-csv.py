# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:45:29 2022

@author: DELL
"""

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers


### In memory data
# load it into memory as a pandas Dataframe or a NumPy array
abalone_train = pd.read_csv(
    # "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    "./abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

# abalone_train.head()

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

# regression model
abalone_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

# abalone_model.fit(abalone_features, abalone_labels, epochs=10)

### Basic preprocessing
normalize = layers.Normalization()
normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                           optimizer = tf.optimizers.Adam())

# norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)


#### Mixed data types
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
# titanic.head()

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

# build a model that implements the preprocessing logic using Keras functional API

# # Create a symbolic input
# input = tf.keras.Input(shape=(), dtype=tf.float32)

# # Perform a calculation using the input
# result = 2*input + 1

# # the result doesn't have a value
# # result

# calc = tf.keras.Model(inputs=input, outputs=result)

# print(calc(1).numpy())
# print(calc(2).numpy())

#  building a set of symbolic keras.Input objects
inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# concatenate the numeric inputs together, and run them through a normalization layer
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

# Collect all the symbolic preprocessing results, to concatenate them later
preprocessed_inputs = [all_numeric_inputs]

# For the string inputs use the tf.keras.layers.StringLookup function to map 
# from strings to integer indices in a vocabulary. Next, use 
# tf.keras.layers.CategoryEncoding to convert the indexes into float32 data 
# appropriate for the model.
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = layers.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

# concatenate all the preprocessed inputs together, and build a model that 
# handles the preprocessing
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

# draw in console
# tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=96, show_shapes=True)

## convert it to a dictionary of tensors
titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}

# Slice out the first training example and pass it to this preprocessing model,
features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
# titanic_preprocessing(features_dict)

# build model
def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)
# train
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

# save the model and reload it
titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')


features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)


#########tf.data

# On in memory data
# manually slice up the dictionary of features from the previous section.
# For each index, it takes that index for each feature
import itertools

def slices(features):
  for i in itertools.count():
    # For each feature take index `i`
    example = {name:values[i] for name, values in features.items()}
    yield example

for example in slices(titanic_features_dict):
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break

# the Dataset.from_tensor_slices
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)
for example in features_ds:
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break

# makes a dataset of (features_dict, labels) pairs
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
# shuffle and batch the data
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)
# nstead of passing features and labels to Model.fit, you pass the datasets
titanic_model.fit(titanic_batches, epochs=5)

##### From a single file
titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

# read the CSV data from the file and create a tf.data.Dataset
titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size=5, # Artificially small to make examples easier to show.
    label_name='survived',
    num_epochs=1,
    ignore_errors=True,)

for batch, label in titanic_csv_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value}")
  print()
  print(f"{'label':20s}: {label}")


# decompress the data on the fly
traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz', 
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir='.', cache_subdir='traffic')

traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size=256,
    label_name='traffic_volume',
    num_epochs=1,
    compression_type="GZIP")

for batch, label in traffic_volume_csv_gz_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value[:5]}")
  print()
  print(f"{'label':20s}: {label[:5]}")

## Caching
# use Dataset.cache or data.experimental.snapshot so that the csv data is only parsed on the first epoch
# cache files can only be used by the TensorFlow process that created them, but snapshot files can be read by other processes
import time

# timeline = []
# timeline.append(time.time())

# # no cache
# for i, (batch, label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
#   if i % 40 == 0:
#     print('.', end='')
# print()
# timeline.append(time.time())
# print(timeline[1] - timeline[0])
# timeline.pop(0)

# # cache
# caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)

# for i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):
#   if i % 40 == 0:
#     print('.', end='')
# print()

# timeline.append(time.time())
# print(timeline[1] - timeline[0])
# timeline.pop(0)

# # snapshot
# snapshot = tf.data.experimental.snapshot('titanic.tfsnap')
# snapshotting = traffic_volume_csv_gz_ds.apply(snapshot).shuffle(1000)

# for i, (batch, label) in enumerate(snapshotting.shuffle(1000).repeat(20)):
#   if i % 40 == 0:
#     print('.', end='')
# print()

# timeline.append(time.time())
# print(timeline[1] - timeline[0])


#### Multiple files
##  character font images dataset
# fonts_zip = tf.keras.utils.get_file(
#     'fonts.zip',  "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
#     cache_dir='.', cache_subdir='fonts',
#     extract=True)

import pathlib
font_csvs =  sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

# When dealing with a bunch of files you can pass a glob-style file_pattern 
# to the experimental.make_csv_dataset function.
# Use the num_parallel_reads argument to set how many files are read in 
# parallel and interleaved together

fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)

# view
for features in fonts_ds.take(1):
  for i, (name, value) in enumerate(features.items()):
    if i>15:
      break
    print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")

# pack the pixels into an image-tensor
import re

def make_images(features):
  image = [None]*400
  new_feats = {}

  for name, value in features.items():
    match = re.match('r(\d+)c(\d+)', name)
    if match:
      image[int(match.group(1))*20+int(match.group(2))] = value
    else:
      new_feats[name] = value

  image = tf.stack(image, axis=0)
  image = tf.reshape(image, [20, 20, -1])
  new_feats['image'] = image

  return new_feats

# Apply that function to each batch in the dataset
fonts_image_ds = fonts_ds.map(make_images)

for features in fonts_image_ds.take(1):
  break

# plot
from matplotlib import pyplot as plt

plt.figure(figsize=(6,6), dpi=120)

for n in range(9):
  plt.subplot(3,3,n+1)
  plt.imshow(features['image'][..., n])
  plt.title(chr(features['m_label'][n]))
  plt.axis('off')
  

### Lower level functions

# tf.io.decode_csv