#!/usr/bin/env python3

import os
from classify_dataset import evaluate, img_width, img_height, N_CLASSES, N_CHANNELS
from dataset_utilities import get_weights_dict
from keras_video import VideoFrameGenerator
import tensorflow as tf

# make sure to provide correct paths to the folders on your machine
data_dir = '../kaggle-dataset-6classes/videos/trainval'
test_data_dir = '../kaggle-dataset-6classes/videos/test'

BATCH_SIZE = 4

FPS = 30

CLASS_NAMES = [str(i) for i in range(N_CLASSES)]

# how many frames to concatenate as input to the network
N_FRAMES = 5

# for data augmentation
data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=60) # 60 degrees

train_ds = VideoFrameGenerator(
    classes=CLASS_NAMES,
    glob_pattern=os.path.join(data_dir, '{classname}/*.mp4'),
    nb_frames=N_FRAMES,
    split_val=0.2,
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_shape=(img_width, img_height),
    nb_channel=N_CHANNELS,
    rescale=1,
    frame_step = FPS // N_FRAMES,
    transformation=data_aug,
    use_frame_cache=False)

val_ds = train_ds.get_validation_generator()

test_ds = VideoFrameGenerator(
    classes=CLASS_NAMES,
    glob_pattern=os.path.join(test_data_dir, '{classname}/*.mp4'),
    nb_frames=N_FRAMES,
    shuffle=False,
    batch_size=BATCH_SIZE,
    target_shape=(img_width, img_height),
    nb_channel=N_CHANNELS,
    rescale=1,
    frame_step = FPS // N_FRAMES,
    transformation=data_aug,
    use_frame_cache=False)


weights_dict = get_weights_dict(data_dir, CLASS_NAMES)

evaluate("kaggle-videos", train_ds, val_ds, test_ds, weights_dict)
