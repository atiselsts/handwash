#!/usr/bin/env python3

import os
from classify_dataset import evaluate, get_time_distributed_model, IMG_SIZE, N_CLASSES
from dataset_utilities import get_weights_dict
import tensorflow as tf
from generator_timedistributed import timedistributed_dataset_from_directory

# make sure to provide correct paths to the folders on your machine
data_dir = '../kaggle-dataset-6classes/videos/trainval'
test_data_dir = '../kaggle-dataset-6classes/videos/test'

BATCH_SIZE = 4

FPS = 30

CLASS_NAMES = [str(i) for i in range(N_CLASSES)]

# how many frames to concatenate as input to the network
N_FRAMES = 5

train_ds = timedistributed_dataset_from_directory(
    data_dir,
    num_frames=N_FRAMES,
    frame_step=FPS // N_FRAMES,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    shuffle=True,
    label_mode='categorical',
    batch_size=BATCH_SIZE)

val_ds = timedistributed_dataset_from_directory(
    data_dir,
    num_frames=N_FRAMES,
    frame_step=FPS // N_FRAMES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    shuffle=True,
    label_mode='categorical',
    batch_size=BATCH_SIZE)

test_ds = timedistributed_dataset_from_directory(
    test_data_dir,
    num_frames=N_FRAMES,
    frame_step=FPS // N_FRAMES,
    seed=123,
    image_size=IMG_SIZE,
    shuffle=False,
    label_mode='categorical',
    batch_size=BATCH_SIZE)

weights_dict = get_weights_dict(data_dir, CLASS_NAMES)

model = get_time_distributed_model(N_FRAMES)

evaluate("kaggle-videos", train_ds, val_ds, test_ds, weights_dict, model=model)
