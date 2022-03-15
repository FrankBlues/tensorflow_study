# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:10:35 2022

@author: DELL
"""

import tensorflow as tf

# Helper libraries
import numpy as np
import os

print(tf.__version__)

global_batch_size = 16
# Create a tf.data.Dataset object.
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)

@tf.function
def train_step(inputs):
  features, labels = inputs
  return labels - 0.3 * features

# Iterate over the dataset using the for..in construct.
for inputs in dataset:
  print(train_step(inputs))
  
#### tf.distribute.Strategy.experimental_distribute_dataset
global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)
# Distribute input using the `experimental_distribute_dataset`.
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
# 1 global batch of data fed to the model in 1 step.
print(next(iter(dist_dataset)))


#### tf.distribute.Strategy.distribute_datasets_from_function
mirrored_strategy = tf.distribute.MirroredStrategy()

def dataset_fn(input_context):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(64).batch(16)
  dataset = dataset.shard(
    input_context.num_input_pipelines, input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2) # This prefetches 2 batches per device.
  return dataset

dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)

