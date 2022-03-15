# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:16:09 2022
  * High level keras model.save and tf.keras.models.load_model
  * Low level tf.saved_model.save and tf.saved_model.load

@author: DELL
"""

import tensorflow_datasets as tfds

import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
  datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  BUFFER_SIZE = 10000

  BATCH_SIZE_PER_REPLICA = 64
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

  train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

  return train_dataset, eval_dataset

def get_model():
  with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model
  
model = get_model()
train_dataset, eval_dataset = get_data()
model.fit(train_dataset, epochs=2)

# save
keras_model_path = "/tmp/keras_save"
model.save(keras_model_path)
# restore
restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs=2)

# load the model and train it using a tf.distribute.Strategy
another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
with another_strategy.scope():
  restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
  restored_keras_model_ds.fit(train_dataset, epochs=2)
  
####  tf.saved_model APIs
model = get_model()  # get a fresh model
saved_model_path = "/tmp/tf_save"
tf.saved_model.save(model, saved_model_path)
#  it returns an object that contain functions that can be used to do inference
DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

# "serving_default" is the default key for the inference function with a saved Keras model
predict_dataset = eval_dataset.map(lambda image, label: image)
for batch in predict_dataset.take(1):
  print(inference_func(batch))
  
# load and do inference in a distributed manner
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
  loaded = tf.saved_model.load(saved_model_path)
  inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

  dist_predict_dataset = another_strategy.experimental_distribute_dataset(
      predict_dataset)

  # Calling the function in a distributed manner
  for batch in dist_predict_dataset:
    another_strategy.run(inference_func,args=(batch,))

# continue training.  TF Hub has hub.KerasLayer for this purpose
import tensorflow_hub as hub

def build_model(loaded):
  x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
  # Wrap what's loaded to a KerasLayer
  keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
  model = tf.keras.Model(x, keras_layer)
  return model

another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
  loaded = tf.saved_model.load(saved_model_path)
  model = build_model(loaded)

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
  model.fit(train_dataset, epochs=2)

### save a Keras model with model.save, and load a non-Keras model with the low-level API
model = get_model()

# Saving the model using Keras's save() API
model.save(keras_model_path) 

another_strategy = tf.distribute.MirroredStrategy()
# Loading the model using lower level API
with another_strategy.scope():
  loaded = tf.saved_model.load(keras_model_path)
  
### Saving/Loading from local device
model = get_model()

# Saving the model to a path on localhost.
saved_model_path = "/tmp/tf_save"
save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
model.save(saved_model_path, options=save_options)

# Loading the model from a path on localhost.
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
  load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
  loaded = tf.keras.models.load_model(saved_model_path, options=load_options)
