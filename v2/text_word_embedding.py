# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:32:46 2022

  * Representing text as numbers
    - One-hot encodings  
      % inefficient
    - Encode each word with a unique number  
      % does not capture any relationship between words
    - Word embeddings  
      % An embedding is a dense vector of floating point values
      %  they are trainable parameters
      % 8-dimensional (for small datasets), up to 1024-dimensions

*** Download from colab
from google.colab import files
files.download('vectors.tsv')

@author: DELL
"""

import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

### Download the IMDb Dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
#                                   untar=True
#                                  )

# dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
dataset_dir = r'C:\Users\DELL\.keras\datasets\aclImdb'
train_dir =r"C:\Users\DELL\.keras\datasets\aclImdb\train"
# os.listdir(dataset_dir)

# dataset
batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

# view
for text_batch, label_batch in train_ds.take(1):
  for i in range(5):
    print(label_batch[i].numpy(), text_batch.numpy()[i])

### Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

### Using the Embedding layer
# Embed a 1,000 word vocabulary into 5 dimensions.
# embedding_layer = tf.keras.layers.Embedding(1000, 5)

# # view
# result = embedding_layer(tf.constant([1, 2, 3]))
# print(result.numpy())

# result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
# print(result.shape)

###Text preprocessing
# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 250

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


### Create a classification model
embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  # tf.keras.layers.Dropout(0.2),
  GlobalAveragePooling1D(),
  # tf.keras.layers.Dropout(0.2),
  Dense(16, activation='relu'),
  Dense(1)
])



### Compile and train the model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

model.summary()

### Retrieve the trained word embeddings and save them to disk
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# Write the weights to disk
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()



