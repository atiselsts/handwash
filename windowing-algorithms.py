#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path

import pathlib
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

from sklearn.metrics import classification_report, confusion_matrix

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load the model
#path_to_saved_model = "all-classification-MobileNetV2.20-0.62.h5"
#path_to_saved_model = "../VPP_Exp11.10-0.79.h5"
path_to_saved_model = "../VPP_Exp14.10-0.90.h5"
saved_model = load_model(path_to_saved_model)
saved_model.trainable = False

# make sure to provide correct paths to the folders on your machine
data_dir = pathlib.Path('../ansis-main-repo/data/frames/trainval')
test_data_dir = pathlib.Path('../ansis-main-repo/data/frames/test')
mitc_data_dir = pathlib.Path('../RSU_MITC_trainval_test/test')
mitc_single_data_dir = pathlib.Path('/home/atis/work/hands/RSU_MITC/1.interfeiss/2021-06-30_11-18-05-3-1/a')

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
# 16 works well on most GPU
# 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
batch_size = 16
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
IMG_SHAPE = IMG_SIZE + (3,)

#test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#  test_data_dir,
#  seed=123,
#  image_size=IMG_SIZE,
#  label_mode='categorical',
#  batch_size=batch_size)

mitc_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  mitc_data_dir,
  seed=123,
  validation_split=0.01,
  subset="validation",
  image_size=IMG_SIZE,
  label_mode='categorical',
  batch_size=batch_size)

#mitc_mini_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#  mitc_single_data_dir,
#  seed=123,
#  image_size=IMG_SIZE,
#  label_mode='categorical',
#  batch_size=batch_size)


#mitc_mini_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#  mitc_single_data_dir,
#  seed=123,
#  image_size=IMG_SIZE,
#  label_mode='categorical',
#    shuffle=False,
#    batch_size=1)


def evaluate_all_experiments():
    y_true = []
    y_predicted = []
    window_t = []
    window_p = []
    
    topdir = '/home/atis/work/hands/RSU_MITC_onlytest/'
    n = 0
    for subdir1 in os.listdir(topdir):
        for subdir2 in os.listdir(os.path.join(topdir, subdir1)):
            f = os.path.join(topdir, subdir1, subdir2, "phone")
            print(f)
            try:
             ds = tf.keras.preprocessing.image_dataset_from_directory(
                f,
                seed=123,
                image_size=IMG_SIZE,
                label_mode='categorical',
                shuffle=False,
                batch_size=1)
            except:
                continue
            yt, yp, wt, wp = measure_performance(saved_model, ds, f)
            y_true += yt
            y_predicted += yp
            window_t += wt
            window_p += wp
            n += 1
#            if n > 2:
#                break
        break
    
    print("non-windowed")
    print_scores(y_true, y_predicted)
    
    print("windowed")
    print_scores(window_t, window_p)



# check the names of the classes
class_names = mitc_dataset.class_names
print(class_names)
num_classes = len(class_names)

def get_frame_number(filename):
    filename = os.path.splitext(os.path.basename(filename))[0]
    underscore = filename.rfind("_")
    if underscore == -1:
        print("no frame number in ", filename)
        return 0
    frame_num = filename[underscore+1:]
    return int(frame_num)


def measure_performance(model, ds, filename=None):
    y_true = []
    y_predicted = []
    n = 0
    for batch in ds:
        b1, b2 = batch
        #print(b1)
        predicted = model.predict(b1)
        for y_pred, y_t in zip(predicted, b2):
            y_predicted.append(int(np.argmax(y_pred)))
            y_true.append(int(np.argmax(y_t)))
            n += 1
#            if n % 100 == 0:
#                print(n)
#                break
#        if n % 100 == 0:
#            break

    if filename:
        json_filename = os.path.splitext(filename)[0] + ".json"
        with open(json_filename, "w") as outf:
            outf.write("{\n")
            s = ", ".join([str(u) for u in y_true])
            outf.write('  "true": [' + s  + '],\n')
            s = ", ".join([str(u) for u in y_predicted])
            outf.write('  "predicted": [' + s + ']\n')
            outf.write("}\n")

    ts = [get_frame_number(f) for f in ds.file_paths]
    m = min(ts)
    ts = [u - m for u in ts]
    frame_codes_t = [-1] * (max(ts) + 1)
    frame_codes_p = [-1] * (max(ts) + 1)
    for i, y_t in enumerate(y_true):
        frame_codes_t[ts[i]] = y_t
    for i, y_p in enumerate(y_predicted):
        frame_codes_p[ts[i]] = y_p
    window_codes_t = windowize_ground_truth(frame_codes_t)
    window_codes_p = windowize_predicted(frame_codes_p)
    return (y_true, y_predicted, window_codes_t, window_codes_p)


WINDOW_SIZE = 20

def windowize_ground_truth(array):
    output = []
    for i in range(0, len(array), WINDOW_SIZE):
        split = array[i:i + WINDOW_SIZE]
        first = split[0]
        out = -1
        # accept the window only if all elements match
        if all([u == first for u in split]):
            out = first
        output.append(out)
    return output


def most_frequent(lst):
    counts = [0] * num_classes
    for el in lst:
        counts[int(el)] += 1
    best = 0
    counts[0] /= 3 # penalize the class 0 a lot more
    for i in range(1, num_classes):
        if counts[best] < counts[i]:
            best = i
    return best


def windowize_predicted(array):
    output = []
    for i in range(0, len(array), WINDOW_SIZE):
        split = array[i:i + WINDOW_SIZE]
        out = most_frequent(split)
        output.append(out)
    return output


def print_scores(y_true, y_predicted):
    matrix = [[0] * num_classes for i in range(num_classes)]
    for actual, predicted in zip(y_true, y_predicted):
        if actual == -1:
            continue
        matrix[actual][predicted] += 1
    
    print("Confusion matrix:")
    for row in matrix:
        print(row)
    
    f1_scores = []
    for i in range(num_classes):
        total = sum(matrix[i])
        true_predictions = matrix[i][i]
        total_predictions = sum([matrix[j][i] for j in range(num_classes)])
        if total:
            precision = true_predictions / total
        else:
            precision = 0
        if total_predictions:
            recall = true_predictions / total_predictions
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        print("{} precision={:.2f}% recall={:.2f}% f1={:.2f}".format(i, 100 * precision, 100 * recall, f1))
        f1_scores.append(f1)
    print("Average F1 score: {:.2f}".format(np.mean(f1_scores)))



#print(saved_model.summary())

#print("\nUsing test data (from hold-out camera locations)")
#measure_performance(saved_model, test_dataset)

#print("\nUsing MITC data")
#measure_performance(saved_model, mitc_dataset)


#print("\nUsing MITC data")
#measure_performance(saved_model, mitc_mini_dataset)

evaluate_all_experiments()
