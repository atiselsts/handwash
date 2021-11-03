#!/usr/bin/env python3

import classify_dataset
import dataset_utilities

# make sure to provide correct paths to the folders on your machine
data_dir = '/data/handwash/kaggle-dataset-6classes-frames/trainval'
test_data_dir = '/data/handwash/kaggle-dataset-6classes-frames/trainval'

train_ds, val_ds, test_ds, weights_dict = get_datasets(data_dir, test_data_dir)

evaluate("kaggle-single-frame", train_ds, val_ds, test_ds, weights_dict)
