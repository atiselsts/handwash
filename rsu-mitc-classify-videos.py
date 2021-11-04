#!/usr/bin/env python3

import os
from classify_dataset import evaluate, get_time_distributed_model, img_width, img_height, N_CLASSES, N_CHANNELS
from dataset_utilities import get_weights_dict
from generator_timedistributed import timedistributed_dataset_from_directory
import tensorflow as tf

# make sure to provide correct paths to the folders on your machine
data_dir = '/data/handwash/RSU_MITC_preprocessed/videos/trainval'
test_data_dir = '/data/handwash/RSU_MITC_preprocessed/videos/test'

BATCH_SIZE = 32

FPS = 16

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

evaluate("rsu-mitc-videos", train_ds, val_ds, test_ds, weights_dict, model=model)
