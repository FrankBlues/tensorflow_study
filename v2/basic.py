# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:21:31 2022

@author: DELL
"""

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data("../MINIST_data")
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  # tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same'),
  # tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
  # tf.keras.layers.Softmax()
])

predictions = model(x_train[:1]).numpy()
print(predictions)

# softmax
print(tf.nn.softmax(predictions).numpy())

# loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
# train
model.fit(x_train, y_train, epochs=5)
# evaluate
model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
# print(probability_model(x_test[:5]))