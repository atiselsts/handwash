#!/usr/bin/env python3

import numpy as np
import os
import json

import matplotlib.pyplot as pl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FRAME_TIME = 0.05 # XXX very approximate

NUM_MOVEMENTS = 7

def evaluate_all_experiments():
    y_true = []
    y_predicted = []
    window_t = []
    window_p = []

    all_done_true = []
    all_done_predicted = []

    durations_true = []
    durations_predicted = []
    
    topdir = '/home/atis/work/hands/RSU_MITC_onlytest/'
    n = 0
    for subdir1 in os.listdir(topdir):
        for subdir2 in os.listdir(os.path.join(topdir, subdir1)):
            f = os.path.join(topdir, subdir1, subdir2, "phone.json")
            print(f)
            with open(f, "r") as inf:
                obj = json.load(inf)
                yt = obj["true"]
                yp = obj["predicted"]

            duration_true = 0
            duration_predicted = 0
            movements_done_true = [False] * NUM_MOVEMENTS
            movements_done_predicted = [False] * NUM_MOVEMENTS
            for t, p in zip(yt, yp):
                if t >= 0:
                    movements_done_true[t] = True
                if p >= 0:
                    movements_done_predicted[p] = True
                if t > 0:
                    duration_true += FRAME_TIME
                if p > 0:
                    duration_predicted += FRAME_TIME
            durations_true.append(duration_true)
            durations_predicted.append(duration_predicted)

            all_done_true.append(all(movements_done_true[1:]))
            all_done_predicted.append(all(movements_done_predicted[1:]))

            wt, wp = apply_windowing(yt, yp)
            y_true += yt
            y_predicted += yp
            window_t += wt
            window_p += wp
    
    print("non-windowed")
    print_scores(y_true, y_predicted, "cm-non-windowed.png")
    
    print("\nwindowed")
    print_scores(window_t, window_p, "cm-windowed.png")

#    print("\nall movements done in washing episode?")
#    print_scores(all_done_true, all_done_predicted)

#    durations_errors = [abs(a-b) for a, b in zip(durations_true, durations_predicted)]
#    print("\navg duration error for washing in episode: {:.2f} sec".format(np.mean(durations_errors)))
#    print("mean true washing duration: {:.2f} sec".format(np.mean(durations_true)))



def apply_windowing(y_true, y_predicted):
    window_codes_t = windowize_ground_truth(y_true)

    window_codes_p = windowize_predicted_most_frequent(y_predicted, penalty_for_0=1)
#    window_codes_p = windowize_predicted_most_frequent(y_predicted, penalty_for_0=2)
#    window_codes_p = windowize_predicted_most_frequent(y_predicted, penalty_for_0=3)
#    window_codes_p = windowize_predicted_most_frequent(y_predicted, penalty_for_0=5)
#    window_codes_p = windowize_predicted_median_filter(y_predicted)

    return window_codes_t, window_codes_p


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


#
# Algorithm 1: most frequent element
#
def most_frequent(lst, penalty_for_0):
    counts = [0] * NUM_MOVEMENTS
    for el in lst:
        counts[int(el)] += 1
    best = 0
    counts[0] /= penalty_for_0 # penalize the class 0 a lot more
    for i in range(1, NUM_MOVEMENTS):
        if counts[best] < counts[i]:
            best = i
    return best

def windowize_predicted_most_frequent(array, penalty_for_0):
    output = []
    for i in range(0, len(array), WINDOW_SIZE):
        split = array[i:i + WINDOW_SIZE]
        out = most_frequent(split, penalty_for_0)
        output.append(out)
    return output


#
# Algorithm 2: median filtering with 5 elements
#
def median(arr):
    return sorted(arr)[len(arr) // 2]

def median_filter(data, window_size):
    low = window_size // 2
    high = (window_size + 1) // 2
    n = len(data)
    result = [0] * n
    for i in range(n):
        mn = max(i - low, 0)
        mx = min(i + high, n)
        result[i] = median(data[mn:mx])
    return result

def windowize_predicted_median_filter(array):
    output = []
    for i in range(0, len(array), WINDOW_SIZE):
        split = array[i:i + WINDOW_SIZE]
        out = median(split)
        output.append(out)
    return output


def print_scores(y_true, y_predicted, plot_filename):
    num_classes = max(max(y_true), max(y_predicted)) + 1
    matrix = [[0] * num_classes for i in range(num_classes)]

    y_true_only_valid = []
    y_predicted_only_valid = []

    for actual, predicted in zip(y_true, y_predicted):
        if actual == -1:
            continue
        matrix[actual][predicted] += 1
        y_true_only_valid.append(actual)
        y_predicted_only_valid.append(predicted)

    print("Confusion matrix:")
    for row in matrix:
        print(row)

    # visualize the matrix
    #cm = confusion_matrix(y_true_only_valid, y_predicted_only_valid)
    ConfusionMatrixDisplay.from_predictions(y_true_only_valid, y_predicted_only_valid, normalize="true")
    pl.savefig(plot_filename, format="png" if ".png" in plot_filename else "pdf")

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



evaluate_all_experiments()
