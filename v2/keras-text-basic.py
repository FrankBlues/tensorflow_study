# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:42:05 2022

  * binary classification 
  * text read
  * text preprocessing (TextVectorization)
  * .cache()  .prefetch()
  * text model
  * tf.summary
  * plot of accuracy and loss over time (history)
  * export model for deployment

Using it outside of your model enables you to do asynchronous CPU processing 
and buffering of your data when training on GPU. So, if you're training your 
model on the GPU, you probably want to go with this option to get the best 
performance while developing your model, then switch to including the 
TextVectorization layer inside your model when you're ready to prepare for 
deployment.
  

@author: DELL
"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)

def imdb():
  url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  
  # dataset = tf.keras.utils.get_file("aclImdb_v1", url,
  #                                     untar=True, cache_dir='.',
  #                                     cache_subdir='')
  dataset = r'C:\Users\DELL\.keras\datasets\aclImdb_v1.tar.gz'
  dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
  
  train_dir = os.path.join(dataset_dir, 'train')
  test_dir = os.path.join(dataset_dir, 'test')
  sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
  # with open(sample_file) as f:
  #   print(f.read())
    
  batch_size = 32
  seed = 42
  
  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
      train_dir, 
      batch_size=batch_size, 
      validation_split=0.2, 
      subset='training', 
      seed=seed)
  
  raw_val_ds = tf.keras.utils.text_dataset_from_directory(
      train_dir, 
      batch_size=batch_size, 
      validation_split=0.2, 
      subset='validation', 
      seed=seed)
  
  raw_test_ds = tf.keras.utils.text_dataset_from_directory(
      test_dir, 
      batch_size=batch_size)
  
  #### preprocess
  # Standardization refers to preprocessing the text, typically to remove
  # punctuation or HTML elements to simplify the dataset.
  # Tokenization refers to splitting strings into tokens (for example, 
  # splitting a sentence into individual words, by splitting on whitespace). 
  # Vectorization refers to converting tokens into numbers so they can be 
  # fed into a neural network. 
  def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
  
  max_features = 10000
  sequence_length = 250
  
  vectorize_layer = layers.TextVectorization(
      standardize=custom_standardization,
      max_tokens=max_features,
      output_mode='int',
      output_sequence_length=sequence_length)
  
  # Make a text-only dataset (without labels), then call adapt
  train_text = raw_train_ds.map(lambda x, y: x)
  vectorize_layer.adapt(train_text)
  
  # see
  def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
  
  # retrieve a batch (of 32 reviews and labels) from the dataset
  text_batch, label_batch = next(iter(raw_train_ds))
  first_review, first_label = text_batch[0], label_batch[0]
  print("Review", first_review)
  print("Label", raw_train_ds.class_names[first_label])
  print("Vectorized review", vectorize_text(first_review, first_label))
  
  print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
  print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
  print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
  
  # apply the TextVectorization layer you created earlier to the train, validation, and test dataset
  train_ds = raw_train_ds.map(vectorize_text)
  val_ds = raw_val_ds.map(vectorize_text)
  test_ds = raw_test_ds.map(vectorize_text)
  
  
  ### Configure the dataset for performance
  #  two important methods when loading data to make sure that I/O does not become blocking
  # .cache()  .prefetch()
  
  AUTOTUNE = tf.data.AUTOTUNE
  
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
  
  # model
  embedding_dim = 16
  
  model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])
  
  print(model.summary())
  # 
  model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
  # train
  epochs = 10
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs)
  # evaluate
  loss, accuracy = model.evaluate(test_ds)
  
  print("Loss: ", loss)
  print("Accuracy: ", accuracy)
  
  ### Create a plot of accuracy and loss over time
  history_dict = history.history
  history_dict.keys()
  
  # loss
  acc = history_dict['binary_accuracy']
  val_acc = history_dict['val_binary_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']
  
  epochs = range(1, len(acc) + 1)
  
  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  
  plt.show()
  
  # accuracy
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  
  plt.show()
  
  
  ### export model
  export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
  ])
  
  export_model.compile(
      loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
  )
  
  # Test it with `raw_test_ds`, which yields raw strings
  loss, accuracy = export_model.evaluate(raw_test_ds)
  print(accuracy)
  
  
  ### Inference on new data
  examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
  ]
  
  print(export_model.predict(examples))
  
  ###Including the text preprocessing logic inside your model enables you to export a model for production that simplifies deployment

if __name__ == '__main__':
  dataset = r'C:\Users\DELL\.keras\datasets\aclImdb_v1.tar.gz'
  dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')
  
  train_dir = os.path.join(dataset_dir, 'train')
  test_dir = os.path.join(dataset_dir, 'test')

    
  batch_size = 32
  seed = 42
  
  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
      train_dir, 
      batch_size=batch_size, 
      validation_split=0.2, 
      subset='training', 
      seed=seed)
  
  raw_val_ds = tf.keras.utils.text_dataset_from_directory(
      train_dir, 
      batch_size=batch_size, 
      validation_split=0.2, 
      subset='validation', 
      seed=seed)
  
  raw_test_ds = tf.keras.utils.text_dataset_from_directory(
      test_dir, 
      batch_size=batch_size)
  
  #### preprocess
  # Standardization refers to preprocessing the text, typically to remove
  # punctuation or HTML elements to simplify the dataset.
  # Tokenization refers to splitting strings into tokens (for example, 
  # splitting a sentence into individual words, by splitting on whitespace). 
  # Vectorization refers to converting tokens into numbers so they can be 
  # fed into a neural network. 
  def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
  
  max_features = 10000
  sequence_length = 250
  
  vectorize_layer = layers.TextVectorization(
      standardize=custom_standardization,
      max_tokens=max_features,
      output_mode='int',
      output_sequence_length=sequence_length)
  
  # Make a text-only dataset (without labels), then call adapt
  train_text = raw_train_ds.map(lambda x, y: x)
  vectorize_layer.adapt(train_text)
  
  # see
  def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
  
  
  # apply the TextVectorization layer you created earlier to the train, validation, and test dataset
  train_ds = raw_train_ds.map(vectorize_text)
  val_ds = raw_val_ds.map(vectorize_text)
  test_ds = raw_test_ds.map(vectorize_text)
  
  
  ### Configure the dataset for performance
  #  two important methods when loading data to make sure that I/O does not become blocking
  # .cache()  .prefetch()
  
  AUTOTUNE = tf.data.AUTOTUNE
  
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
  
  # model
  embedding_dim = 8
  
  model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    #layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    #layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)])
  
  print(model.summary())
  # 
  model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=['accuracy'])
  # train
  epochs = 10
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs)
  # evaluate
  loss, accuracy = model.evaluate(test_ds)
  
  print("Loss: ", loss)
  print("Accuracy: ", accuracy)
  
  ### Create a plot of accuracy and loss over time
  history_dict = history.history
  history_dict.keys()
  
  # loss
  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']
  
  epochs = range(1, len(acc) + 1)
  
  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  
  plt.show()
  
  # accuracy
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  
  plt.show()
  
  
  ### export model
  export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
  ])
  
  export_model.compile(
      loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
  )
  
  # Test it with `raw_test_ds`, which yields raw strings
  loss, accuracy = export_model.evaluate(raw_test_ds)
  print(accuracy)
  
  
  ### Inference on new data
  examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
  ]
  
  print(export_model.predict(examples))