#!/usr/bin/env python3

#
# This work is inspired by this NN architecture:
# https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition/blob/master/three_stream_model.py
#

import matplotlib.pyplot as plt
import os
import numpy as np

import tensorflow as tf

#from tensorflow.keras.preprocessing import image_dataset_from_directory
from generator_rgb_with_of import merged_dataset_from_directories
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# make sure to provide correct paths to the folders on your machine
rgb_dir = '../kaggle-dataset-6classes-frames'
of_dir = '../kaggle-dataset-6classes-of'

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
# 16 works well on most GPU
# 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
batch_size = 16
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
IMG_SHAPE = IMG_SIZE + (3,)

N_CLASSES = 7


# data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])



# rescale pixel values
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
for layer in base_model.layers:
    layer.trainable = False

# Build the model
rgb_network_input = tf.keras.Input(shape=IMG_SHAPE)
rgb_network = data_augmentation(rgb_network_input)
rgb_network = preprocess_input(rgb_network)
rgb_network = base_model(rgb_network, training=False)
rgb_network = tf.keras.layers.Flatten()(rgb_network)
rgb_network = tf.keras.Model(rgb_network_input, rgb_network)

for layer in rgb_network.layers:
    layer._name = "rgb_" + layer.name

of_network_input = tf.keras.Input(shape=IMG_SHAPE)
of_network = data_augmentation(of_network_input)
of_network = preprocess_input(of_network)
of_network = base_model(of_network, training=False)
of_network = tf.keras.layers.Flatten()(of_network)
of_network = tf.keras.Model(of_network_input, of_network)

for layer in of_network.layers:
    layer._name = "of_" + layer.name

merged = tf.keras.layers.concatenate([rgb_network.output, of_network.output], axis=1)
merged = tf.keras.layers.Flatten()(merged)
#merged = tf.keras.layers.Dense(64, activation='relu')(merged)
merged = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(merged)

model = tf.keras.Model([rgb_network.input, of_network.input], merged)

print(model.summary())

print("compiling the model...")
model.compile(optimizer='SGD',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.save("test.h5")

number_of_epochs = 10

# callbacks to implement early stopping and saving the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint(monitor='val_accuracy', mode='max',
                     verbose=1, save_freq='epoch',
                     filepath='kaggle-single-frame-merged-of.{epoch:02d}-{val_accuracy:.2f}.h5')


train_ds = merged_dataset_from_directories(
  rgb_dir,
  of_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=IMG_SIZE,
  shuffle=True,
  label_mode='categorical',
  batch_size=batch_size)

val_ds = merged_dataset_from_directories(
  rgb_dir,
  of_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=IMG_SIZE,
  shuffle=True,
  label_mode='categorical',
  batch_size=batch_size)

# check the names of the classes
class_names = train_ds.class_names
print(class_names)

# As the dataset is imbalanced, is is necessary to get weights for each class
# get the number of trainval images for each class
images_by_labels = []
for i in range(len(class_names)):
    for subdir, dirs, files in os.walk(os.path.join(of_dir,str(i))):
        n_of_files = 0
        for image_file in files:
            if image_file.endswith("jpg"):
                n_of_files += 1
        images_by_labels.append(n_of_files)

# calculate weights
images_by_labels = np.array(images_by_labels)
avg = np.average(images_by_labels)
weights = avg / images_by_labels

# create dictionary with weights as required for keras fit() function
weights_dict = {}
for item in range(len(weights)):
    weights_dict[int(class_names[item])] = weights[item]
print("weights_dict=", weights_dict)

# to improve performance, use buffered prefetching to load images
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


print("fitting the model...")
history = model.fit(train_ds,
                    epochs=number_of_epochs,
                    validation_data=val_ds,
                    class_weight=weights_dict,
                    callbacks=[es, mc])

# visualise accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.savefig("accuracy.pdf", format="pdf")
