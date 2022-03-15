# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:20:47 2022

@author: DELL
"""

import multiprocessing
import os
import random
import portpicker
import tensorflow as tf

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

# Instantiate a ParameterServerStrategy
variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

# Variable sharding

### Training with Model.fit
## Input data
def dataset_fn(input_context):
  global_batch_size = 64
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)

  x = tf.random.uniform((10, 10))
  y = tf.random.uniform((10,))

  dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
  dataset = dataset.shard(
      input_context.num_input_pipelines,
      input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)

  return dataset

dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

## Model construction and compiling
with strategy.scope():
  model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

  model.compile(tf.keras.optimizers.SGD(), loss='mse', steps_per_execution=10)

## Callbacks and training
working_dir = '/tmp/my_working_dir'
log_dir = os.path.join(working_dir, 'log')
ckpt_filepath = os.path.join(working_dir, 'ckpt')
backup_dir = os.path.join(working_dir, 'backup')

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
]

model.fit(dc, epochs=5, steps_per_epoch=20, callbacks=callbacks)

feature_vocab = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong", "wonder_woman"
]
label_vocab = ["yes", "no"]

with strategy.scope():
  feature_lookup_layer = tf.keras.layers.StringLookup(
      vocabulary=feature_vocab,
      mask_token=None)
  label_lookup_layer = tf.keras.layers.StringLookup(
      vocabulary=label_vocab,
      num_oov_indices=0,
      mask_token=None)

  raw_feature_input = tf.keras.layers.Input(
      shape=(3,),
      dtype=tf.string,
      name="feature")
  feature_id_input = feature_lookup_layer(raw_feature_input)
  feature_preprocess_stage = tf.keras.Model(
      {"features": raw_feature_input},
      feature_id_input)

  raw_label_input = tf.keras.layers.Input(
      shape=(1,),
      dtype=tf.string,
      name="label")
  label_id_input = label_lookup_layer(raw_label_input)

  label_preprocess_stage = tf.keras.Model(
      {"label": raw_label_input},
      label_id_input)

# Generate toy examples in a dataset
def feature_and_label_gen(num_examples=200):
  examples = {"features": [], "label": []}
  for _ in range(num_examples):
    features = random.sample(feature_vocab, 3)
    label = ["yes"] if "avenger" in features else ["no"]
    examples["features"].append(features)
    examples["label"].append(label)
  return examples

examples = feature_and_label_gen()

# create the training dataset wrapped in a dataset_fn
def dataset_fn(_):
  raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

  train_dataset = raw_dataset.map(
      lambda x: (
          {"features": feature_preprocess_stage(x["features"])},
          label_preprocess_stage(x["label"])
      )).shuffle(200).batch(32).repeat()
  return train_dataset

## Build the model
# These variables created under the `strategy.scope` will be placed on parameter
# servers in a round-robin fashion.
with strategy.scope():
  # Create the model. The input needs to be compatible with Keras processing layers.
  model_input = tf.keras.layers.Input(
      shape=(3,), dtype=tf.int64, name="model_input")

  emb_layer = tf.keras.layers.Embedding(
      input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
  emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
  dense_output = tf.keras.layers.Dense(units=1, activation="sigmoid")(emb_output)
  model = tf.keras.Model({"features": model_input}, dense_output)

  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
  accuracy = tf.keras.metrics.Accuracy()

# Let's confirm that the use of FixedShardsPartitioner split all variables into two shards and each shard was assigned to different parameter servers

assert len(emb_layer.weights) == 2
assert emb_layer.weights[0].shape == (4, 16384)
assert emb_layer.weights[1].shape == (4, 16384)
assert emb_layer.weights[0].device == "/job:ps/replica:0/task:0/device:CPU:0"
assert emb_layer.weights[1].device == "/job:ps/replica:0/task:1/device:CPU:0"

## Define the training step
#  create the training step wrapped into a tf.function
@tf.function
def step_fn(iterator):

  def replica_fn(batch_data, labels):
    with tf.GradientTape() as tape:
      pred = model(batch_data, training=True)
      per_example_loss = tf.keras.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
      loss = tf.nn.compute_average_loss(per_example_loss)
      gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    accuracy.update_state(labels, actual_pred)
    return loss

  batch_data, labels = next(iterator)
  losses = strategy.run(replica_fn, args=(batch_data, labels))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


## Dispatch training steps to remote workers
# create a ClusterCoordinator object and pass in the strategy object
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

# create a per-worker dataset and an iterator
@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)

#  distribute the computation to remote workers using ClusterCoordinator.schedule
num_epoches = 4
steps_per_epoch = 5
for i in range(num_epoches):
  accuracy.reset_states()
  for _ in range(steps_per_epoch):
    coordinator.schedule(step_fn, args=(per_worker_iterator,))
  # Wait at epoch boundaries.
  coordinator.join()
  print ("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

#  fetch the result of a RemoteValue
loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
print ("Final loss is %f" % loss.fetch())

# for _ in range(total_steps):
#   coordinator.schedule(step_fn, args=(per_worker_iterator,))
# while not coordinator.done():
#   time.sleep(10)
  # Do something like logging metrics or writing checkpoints.
  
#### Evaluation
## Inline evaluation
# Direct evaluation
eval_dataset = tf.data.Dataset.from_tensor_slices(
      feature_and_label_gen(num_examples=16)).map(
          lambda x: (
              {"features": feature_preprocess_stage(x["features"])},
              label_preprocess_stage(x["label"])
          )).batch(8)

eval_accuracy = tf.keras.metrics.Accuracy()

for batch_data, labels in eval_dataset:
  pred = model(batch_data, training=False)
  actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
  eval_accuracy.update_state(labels, actual_pred)

print ("Evaluation accuracy: %f" % eval_accuracy.result())

# Distributed evaluation
with strategy.scope():
  # Define the eval metric on parameter servers.
  eval_accuracy = tf.keras.metrics.Accuracy()

@tf.function
def eval_step(iterator):
  def replica_fn(batch_data, labels):
    pred = model(batch_data, training=False)
    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    eval_accuracy.update_state(labels, actual_pred)
  batch_data, labels = next(iterator)
  strategy.run(replica_fn, args=(batch_data, labels))

def eval_dataset_fn():
  return tf.data.Dataset.from_tensor_slices(
      feature_and_label_gen(num_examples=16)).map(
          lambda x: (
              {"features": feature_preprocess_stage(x["features"])},
              label_preprocess_stage(x["label"])
          )).shuffle(16).repeat().batch(8)

per_worker_eval_dataset = coordinator.create_per_worker_dataset(eval_dataset_fn)
per_worker_eval_iterator = iter(per_worker_eval_dataset)

eval_steps_per_epoch = 2
for _ in range(eval_steps_per_epoch):
  coordinator.schedule(eval_step, args=(per_worker_eval_iterator,))
coordinator.join()
print ("Evaluation accuracy: %f" % eval_accuracy.result())

## Side-car evaluation
checkpoint_dir = ...
eval_model = ...
eval_data = ...
checkpoint = tf.train.Checkpoint(model=eval_model)

for latest_checkpoint in tf.train.checkpoints_iterator(
    checkpoint_dir):
  try:
    checkpoint.restore(latest_checkpoint).expect_partial()
  except (tf.errors.OpError,) as e:
    # checkpoint may be deleted by training when it is about to read it.
    continue

  # Optionally add callbacks to write summaries.
  eval_model.evaluate(eval_data)

  # Evaluation finishes when it has evaluated the last epoch.
  # if latest_checkpoint.endswith('-{}'.format(train_epoches)):
  #   break