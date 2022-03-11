# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:02:45 2022

  * tf.keras.utils.text_dataset_from_directory
  * tf.keras.layers.TextVectorization
  * tf.data.TextLineDataset
  * tf.lookup.StaticVocabularyTable
  * UnicodeScriptTokenizer
  * pad
  * set_vocabulary
@author: DELL
"""

import collections
import pathlib

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text


VOCAB_SIZE = 10000

# int mode 
MAX_SEQUENCE_LENGTH = 250

BATCH_SIZE = 64

AUTOTUNE = tf.data.AUTOTUNE
  
def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

# use the 'int' vectorized layer to build a 1D ConvNet
def create_model(vocab_size, num_labels):
  model = tf.keras.Sequential([
      layers.Embedding(vocab_size, 64, mask_zero=True),
      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
      layers.GlobalMaxPooling1D(),
      layers.Dense(num_labels)
  ])
  return model

##### Example 1: Predict the tag for a Stack Overflow question
def example1():
  ## download and explore data
  data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
  
  dataset_dir = utils.get_file(
      origin=data_url,
      untar=True,
      )
  
  dataset_dir = pathlib.Path(dataset_dir).parent
  train_dir = dataset_dir/'stack_overflow_16k'/'train'
  
  # load
  batch_size = 32
  seed = 42
  
  raw_train_ds = utils.text_dataset_from_directory(
      train_dir,
      batch_size=batch_size,
      validation_split=0.2,
      subset='training',
      seed=seed)
  
  for i, label in enumerate(raw_train_ds.class_names):
    print("Label", i, "corresponds to", label)
  
  #### specify a random seed or pass shuffle=False, so that the validation and training splits have no overlap
  # Create a validation set.
  raw_val_ds = utils.text_dataset_from_directory(
      train_dir,
      batch_size=batch_size,
      validation_split=0.2,
      subset='validation',
      seed=seed)
  
  test_dir = dataset_dir/'stack_overflow_16k'/'test'
  
  # Create a test set.
  raw_test_ds = utils.text_dataset_from_directory(
      test_dir,
      batch_size=batch_size)
  
  
  ###Prepare the dataset for training
  
  # use the 'binary' vectorization mode to build a bag-of-words model
  # use the 'int' mode with a 1D ConvNet
  VOCAB_SIZE = 10000
  
  binary_vectorize_layer = TextVectorization(
      max_tokens=VOCAB_SIZE,
      output_mode='binary')
  
  # int mode 
  MAX_SEQUENCE_LENGTH = 250
  
  int_vectorize_layer = TextVectorization(
      max_tokens=VOCAB_SIZE,
      output_mode='int',
      output_sequence_length=MAX_SEQUENCE_LENGTH)
  
  # Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
  train_text = raw_train_ds.map(lambda text, labels: text)
  binary_vectorize_layer.adapt(train_text)
  int_vectorize_layer.adapt(train_text)
  
  def binary_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return binary_vectorize_layer(text), label
  
  def int_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return int_vectorize_layer(text), label
  
  # Retrieve a batch (of 32 reviews and labels) from the dataset.
  text_batch, label_batch = next(iter(raw_train_ds))
  first_question, first_label = text_batch[0], label_batch[0]
  print("Question", first_question)
  print("Label", first_label)
  
  print("'binary' vectorized question:",
        binary_vectorize_text(first_question, first_label)[0])
  
  print("'int' vectorized question:",
        int_vectorize_text(first_question, first_label)[0])
  
  binary_train_ds = raw_train_ds.map(binary_vectorize_text)
  binary_val_ds = raw_val_ds.map(binary_vectorize_text)
  binary_test_ds = raw_test_ds.map(binary_vectorize_text)
  
  int_train_ds = raw_train_ds.map(int_vectorize_text)
  int_val_ds = raw_val_ds.map(int_vectorize_text)
  int_test_ds = raw_test_ds.map(int_vectorize_text)
  
  
  # configure dataset
  binary_train_ds = configure_dataset(binary_train_ds)
  binary_val_ds = configure_dataset(binary_val_ds)
  binary_test_ds = configure_dataset(binary_test_ds)
  
  int_train_ds = configure_dataset(int_train_ds)
  int_val_ds = configure_dataset(int_val_ds)
  int_test_ds = configure_dataset(int_test_ds)
  
  # For the 'binary' vectorized data, define a simple bag-of-words linear model, then configure and train i
  binary_model = tf.keras.Sequential([layers.Dense(4)])
  
  binary_model.compile(
      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=['accuracy'])
  
  history = binary_model.fit(
      binary_train_ds, validation_data=binary_val_ds, epochs=10)
  
  
  
  # `vocab_size` is `VOCAB_SIZE + 1` since `0` is used additionally for padding.
  int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4)
  int_model.compile(
      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=['accuracy'])
  history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=10)
  
  print("Linear model on binary vectorized data:")
  print(binary_model.summary())
  
  print("ConvNet model on int vectorized data:")
  print(int_model.summary())
  
  binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
  int_loss, int_accuracy = int_model.evaluate(int_test_ds)
  
  print("Binary model accuracy: {:2.2%}".format(binary_accuracy))
  print("Int model accuracy: {:2.2%}".format(int_accuracy))
  
  ## you can create a new model using the weights you have just trained
  export_model = tf.keras.Sequential(
      [binary_vectorize_layer, binary_model,
       layers.Activation('sigmoid')])
  
  export_model.compile(
      loss=losses.SparseCategoricalCrossentropy(from_logits=False),
      optimizer='adam',
      metrics=['accuracy'])
  
  # Test it with `raw_test_ds`, which yields raw strings
  loss, accuracy = export_model.evaluate(raw_test_ds)
  print("Accuracy: {:2.2%}".format(binary_accuracy))
  
  # Define a function to find the label with the maximum score
  def get_string_labels(predicted_scores_batch):
    predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
    predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
    return predicted_labels
  
  inputs = [
      "how do I extract keys from a dict into a list?",  # 'python'
      "debug public static void main(string[] args) {...}",  # 'java'
  ]
  predicted_scores = export_model.predict(inputs)
  predicted_labels = get_string_labels(predicted_scores)
  for input, label in zip(inputs, predicted_labels):
    print("Question: ", input)
    print("Predicted label: ", label.numpy())


##### Example2 Predict the author of Iliad translations
def example2():
  DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
  FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
  
  for name in FILE_NAMES:
    text_dir = utils.get_file(name, origin=DIRECTORY_URL + name)
  
  parent_dir = pathlib.Path(text_dir).parent
  # list(parent_dir.iterdir())
  
  ## load data
  def labeler(example, index):
    return example, tf.cast(index, tf.int64)
  
  labeled_data_sets = []
  
  for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(str(parent_dir/file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)
  
  # combine these labeled datasets into a single dataset
  BUFFER_SIZE = 50000
  BATCH_SIZE = 64
  VALIDATION_SIZE = 5000
  
  all_labeled_data = labeled_data_sets[0]
  for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
  all_labeled_data = all_labeled_data.shuffle(
      BUFFER_SIZE, reshuffle_each_iteration=False)
  
  for text, label in all_labeled_data.take(10):
    print("Sentence: ", text.numpy())
    print("Label:", label.numpy())
  
  # use the TensorFlow Text APIs to standardize and tokenize the data, build 
  # a vocabulary and use tf.lookup.StaticVocabularyTable to map tokens to 
  # integers to feed to the model
  tokenizer = tf_text.UnicodeScriptTokenizer()
  
  def tokenize(text, unused_label):
    lower_case = tf_text.case_fold_utf8(text)
    return tokenizer.tokenize(lower_case)
  
  tokenized_ds = all_labeled_data.map(tokenize)
  
  for text_batch in tokenized_ds.take(5):
    print("Tokens: ", text_batch.numpy())
  
  # build a vocabulary by sorting tokens by frequency and keeping the top VOCAB_SIZE tokens
  tokenized_ds = configure_dataset(tokenized_ds)
  
  vocab_dict = collections.defaultdict(lambda: 0)
  for toks in tokenized_ds.as_numpy_iterator():
    for tok in toks:
      vocab_dict[tok] += 1
  
  vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
  vocab = [token for token, count in vocab]
  vocab = vocab[:VOCAB_SIZE]
  vocab_size = len(vocab)
  print("Vocab size: ", vocab_size)
  print("First five vocab entries:", vocab[:5])
  
  # map tokens to integers in the range [2, vocab_size + 2]. As with the 
  # TextVectorization layer, 0 is reserved to denote padding and 1 is reserved 
  # to denote an out-of-vocabulary (OOV) token
  keys = vocab
  values = range(2, len(vocab) + 2)  # Reserve `0` for padding, `1` for OOV tokens.
  
  init = tf.lookup.KeyValueTensorInitializer(
      keys, values, key_dtype=tf.string, value_dtype=tf.int64)
  
  num_oov_buckets = 1
  vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
  
  # define a function to standardize, tokenize and vectorize the dataset using the tokenizer and lookup table
  def preprocess_text(text, label):
    standardized = tf_text.case_fold_utf8(text)
    tokenized = tokenizer.tokenize(standardized)
    vectorized = vocab_table.lookup(tokenized)
    return vectorized, label
  
  example_text, example_label = next(iter(all_labeled_data))
  print("Sentence: ", example_text.numpy())
  vectorized_text, example_label = preprocess_text(example_text, example_label)
  print("Vectorized sentence: ", vectorized_text.numpy())
  
  # run the preprocess function on the dataset using Dataset.map
  all_encoded_data = all_labeled_data.map(preprocess_text)
  
  # Split the dataset into training and test sets
  train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
  validation_data = all_encoded_data.take(VALIDATION_SIZE)
  
  train_data = train_data.padded_batch(BATCH_SIZE)
  validation_data = validation_data.padded_batch(BATCH_SIZE)
  
  sample_text, sample_labels = next(iter(validation_data))
  print("Text batch shape: ", sample_text.shape)
  print("Label batch shape: ", sample_labels.shape)
  print("First text example: ", sample_text[0])
  print("First label example: ", sample_labels[0])
  
  vocab_size += 2
  
  train_data = configure_dataset(train_data)
  validation_data = configure_dataset(validation_data)
  
  ### train a model
  model = create_model(vocab_size=vocab_size, num_labels=3)
  
  model.compile(
      optimizer='adam',
      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  
  history = model.fit(train_data, validation_data=validation_data, epochs=3)
  
  loss, accuracy = model.evaluate(validation_data)
  
  print("Loss: ", loss)
  print("Accuracy: {:2.2%}".format(accuracy))
  
  # Export the model
  # you can use TextVectorization.set_vocabulary (instead of 
  # TextVectorization.adapt), which trains a new vocabulary
  preprocess_layer = TextVectorization(
      max_tokens=vocab_size,
      standardize=tf_text.case_fold_utf8,
      split=tokenizer.tokenize,
      output_mode='int',
      output_sequence_length=MAX_SEQUENCE_LENGTH)
  
  preprocess_layer.set_vocabulary(vocab)
  
  export_model = tf.keras.Sequential(
      [preprocess_layer, model,
       layers.Activation('sigmoid')])
  
  export_model.compile(
      loss=losses.SparseCategoricalCrossentropy(from_logits=False),
      optimizer='adam',
      metrics=['accuracy'])
  
  # Create a test dataset of raw strings.
  test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
  test_ds = configure_dataset(test_ds)
  
  loss, accuracy = export_model.evaluate(test_ds)
  
  print("Loss: ", loss)
  print("Accuracy: {:2.2%}".format(accuracy))
  
  ### Run inference on new data
  inputs = [
      "Join'd to th' Ionians with their flowing robes,",  # Label: 1
      "the allies, and his armour flashed about him so that he seemed to all",  # Label: 2
      "And with loud clangor of his arms he fell.",  # Label: 0
  ]
  
  predicted_scores = export_model.predict(inputs)
  predicted_labels = tf.argmax(predicted_scores, axis=1)
  
  for input, label in zip(inputs, predicted_labels):
    print("Question: ", input)
    print("Predicted label: ", label.numpy())

#### Download more datasets using TensorFlow Datasets
# Training set.
train_ds = tfds.load(
    'imdb_reviews',
    split='train[:80%]',
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True)

# Validation set.
val_ds = tfds.load(
    'imdb_reviews',
    split='train[80%:]',
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True)

for review_batch, label_batch in val_ds.take(1):
  for i in range(5):
    print("Review: ", review_batch[i].numpy())
    print("Label: ", label_batch[i].numpy())


### vectorize_layer = TextVectorization(
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

# Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
train_text = train_ds.map(lambda text, labels: text)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)

# Configure datasets for performance as before.
train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)

# Create, configure and train the model
model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=1)
model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=3)

loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))

### Export the model
export_model = tf.keras.Sequential(
    [vectorize_layer, model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy'])

# 0 --> negative review
# 1 --> positive review
inputs = [
    "This is a fantastic movie.",
    "This is a bad movie.",
    "This movie was so bad that it was good.",
    "I will never say yes to watching this movie.",
]

predicted_scores = export_model.predict(inputs)
predicted_labels = [int(round(x[0])) for x in predicted_scores]

for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label)