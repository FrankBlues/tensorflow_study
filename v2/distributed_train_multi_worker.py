# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:36:07 2022

@author: DELL
"""

import json
import os
import sys

## Disable all GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Reset the TF_CONFIG environment variable
os.environ.pop('TF_CONFIG', None)

## Make sure that the current directory is on Python's path
if '.' not in sys.path:
  sys.path.insert(0, '.')
  
import tensorflow as tf

from multi_worker_training import mnist_setup


### Model training on a single worker

batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)


### Multi-worker configuration
# a 'cluster' with several jobs, and each of the jobs may have  one or more 'task's

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

json.dumps(tf_config)

# It is customary for the 'chief' to have 'index' 0 be appointed to

## synchronous multi-worker training
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()

# json-serialize the TF_CONFIG and add it to the environment variables
os.environ['TF_CONFIG'] = json.dumps(tf_config)

##命令行方式

### dataset sharding
## control the sharding by setting the tf.data.experimental.AutoShardPolicy of the tf.data.experimental.DistributeOptions
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

global_batch_size = 64
multi_worker_dataset = mnist_setup.mnist_dataset(batch_size=64)
dataset_no_auto_shard = multi_worker_dataset.with_options(options)


## BackupAndRestore callback
# Multi-worker training with `MultiWorkerMirroredStrategy`
# and the `BackupAndRestore` callback.
# the BackupAndRestore callback supports single-worker training with no 
# strategy—MirroredStrategy—and multi-worker training with MultiWorkerMirroredStrategy
callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup')]
with strategy.scope():
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset,
                       epochs=3,
                       steps_per_epoch=70,
                       callbacks=callbacks)
