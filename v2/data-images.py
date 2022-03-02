# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:23:17 2022

图像打开方式
  * tf.keras.utils.image_dataset_from_directory
  * tf.data.Dataset.list_files
  * tf.io.read_file
  * tf.io.decode_jpeg
  * tf.image.resize

@author: DELL
"""

import numpy as np
import os
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds

print(tf.__version__)


data_dir = "E:/S2/label/"
data_dir = pathlib.Path(data_dir)
# Load data using a Keras utility
batch_size = 16
img_height = 256
img_width = 256

image_count = len(list(data_dir.glob('*/*.jpg')))

# 直接从文件夹读取,每一类位于一个文件夹下
# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# class_names = train_ds.class_names

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# # import matplotlib.pyplot as plt

# # plt.figure(figsize=(10, 10))
# # for images, labels in train_ds.take(1):
# #   for i in range(9):
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(images[i].numpy().astype("uint8"))
# #     plt.title(class_names[labels[i]])
# #     plt.axis("off")

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

# normalization_layer = tf.keras.layers.Rescaling(1./255)

# # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# # image_batch, labels_batch = next(iter(normalized_ds))
# # first_image = image_batch[0]
# # # Notice the pixel values are now in `[0,1]`.
# # print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# model.summary()

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=10
# )

#### tf.data方式读取数据
image_count = 1000
number_per_class = 200
list_ds = None
for item in data_dir.glob('*'):
  print(item.name)
  if item.name != "质量标签.xlsx":
    if list_ds is None:
      list_ds = tf.data.Dataset.list_files(str(data_dir/item.name/'*'), shuffle=True).take(number_per_class)
    else:
      list_ds = list_ds.concatenate(tf.data.Dataset.list_files(str(data_dir/item.name/'*'), shuffle=True).take(number_per_class))

# list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
# list_ds = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=True).take(image_count)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# for f in list_ds.take(5):
#   print(f.numpy())

# class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "质量标签.xlsx"]))
# print(class_names)
class_names = np.array(['1', '2', '3', '4', '5'])

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
  
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)