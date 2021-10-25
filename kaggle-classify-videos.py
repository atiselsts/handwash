#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_video import VideoFrameGenerator

if 1:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

# make sure to provide correct paths to the folders on your machine
data_dir = '../kaggle-dataset-6classes/'

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
# 16 works well on most GPU
# 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
#batch_size = 16
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
N_CHANNELS = 3
IMG_SHAPE = IMG_SIZE + (N_CHANNELS,)

BATCH_SIZE = 4

FPS = 30

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6']
N_CLASSES = len(CLASS_NAMES)

# how many frames to concatenate as input to the network
N_FRAMES = 5

GLOB_PATTERN= os.path.join(data_dir, '{classname}/*.mp4')

# for data augmentation
#data_aug = keras.preprocessing.image.ImageDataGenerator(
#    zoom_range=.1,
#    horizontal_flip=True,
#    rotation_range=8,
#    width_shift_range=.2,
#    height_shift_range=.2)

data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=60) # 60 degrees

train_ds = VideoFrameGenerator(
    classes=CLASS_NAMES, 
    glob_pattern=GLOB_PATTERN,
    nb_frames=N_FRAMES,
    split_val=0.2, 
    shuffle=False,
    batch_size=BATCH_SIZE,
    target_shape=(img_width, img_height),
    nb_channel=N_CHANNELS,
    rescale=1,
    frame_step = FPS // N_FRAMES,
    transformation=data_aug,
    use_frame_cache=False)

val_ds = train_ds.get_validation_generator()

#ds_size = 0
#for batch in train_ds:
#    b1, b2 = batch
#    for b in b1:
#        ds_size += 1

#print("num ds elements:", ds_size)
#exit(0)


#train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#  data_dir,
#  validation_split=0.2,
# subset="training",
#  seed=123,
#  image_size=IMG_SIZE,
#  label_mode='categorical',
#  batch_size=batch_size)

#val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#  data_dir,
#  validation_split=0.2,
#  subset="validation",
#  seed=123,
#  image_size=IMG_SIZE,
#  label_mode='categorical',
#  batch_size=batch_size)

# check the names of the classes
#class_names = train_ds.class_names
#print(class_names)

# As the dataset is imbalanced, is is necessary to get weights for each class
# get the number of trainval images for each class
#images_by_labels = []
#for i in range(len(class_names)):
#    for subdir, dirs, files in os.walk(os.path.join(data_dir,str(i))):
#        n_of_files = 0
#        for image_file in files:
#            if image_file.endswith("jpg"):
#                n_of_files += 1
#        images_by_labels.append(n_of_files)

# calculate weights
#images_by_labels = np.array(images_by_labels)
#avg = np.average(images_by_labels)
#weights = avg / images_by_labels

# create dictionary with weights as required for keras fit() function
#weights_dict = {}
#for item in range(len(weights)):
#    weights_dict[int(class_names[item])] = weights[item]
#print("weights_dict=", weights_dict)


# data augmentation
#data_augmentation = tf.keras.Sequential([
#  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
#])



# rescale pixel values
#preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               pooling='avg',
                                               weights='imagenet')

# freeze the convolutional base
base_model.trainable = False
#trainable = 0
#for layer in base_model.layers[:-trainable]:
#    layer.trainable = False
#for layer in base_model.layers[-trainable:]:
#    layer.trainable = True
for layer in base_model.layers:
    layer.trainable = False

def return_end_model():
    INPUT_SHAPE = (N_FRAMES,) + IMG_SHAPE

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.mobilenet.preprocess_input(inputs)

    encoded_frames = tf.keras.layers.TimeDistributed(base_model)(x)
#    encoded_sequence = tf.keras.layers.LSTM(256)(encoded_frames)
    encoded_sequence = tf.keras.layers.GRU(256)(encoded_frames)

#    hidden_layer = tf.keras.layers.Dense(1024, activation="relu")(encoded_sequence)
#    outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(hidden_layer)

    outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(encoded_sequence)

    model = tf.keras.Model(inputs, outputs)
    return model

model = return_end_model()

#print("compiling the model...")
#base_learning_rate = 0.001 #0.0005 #0.00005
model.compile(optimizer='SGD',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

print(model.summary())

number_of_epochs = 10

# callbacks to implement early stopping and saving the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint(monitor='val_accuracy', mode='max',
                     verbose=1, save_freq='epoch', 
                     filepath='all-classification-sgd-MobileNetV2.{epoch:02d}-{val_accuracy:.2f}.h5')

print("fitting the model...")
history = model.fit(train_ds,
                    epochs=number_of_epochs,
                    validation_data=val_ds,
#                    class_weight=weights_dict,
#                    callbacks=[es, mc]
                    )

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
