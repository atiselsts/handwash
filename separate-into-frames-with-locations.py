#!/usr/bin/env python3

#
# This script takes input folders with video files and annotations
# and converts them to output folders.
# There are two output folders for movement classification:
# * "test" for testing dataset;
# * "trainval" for training and validation data.
# Each of these output folders contains subfolders named "0", "1", ... "7".
# The subfolders have jpeg images of the individual frames.
#

import cv2
import os
import json
import csv
import random

# Change these directories to your own locations
output_folder = './'
input_folder = '../merged_handwash_data'

# movement 0 is large part of the dataset; use only 10% of all frames
MOVEMENT_0_PROPORTION = 0.1

# the movement codes are from 0 to 7
TOTAL_MOVEMENTS = 8

# the Annotator directories go from Annotator1 to Annotator8
TOTAL_ANNOTATORS = 8

# allow up to 1 second for reaction time
REACTION_TIME_FRAMES = 30

FPS = 30

# ignore frames above this duration threshold
MAX_FRAMES = 60 * FPS


def majority_vote(lst):
    """Returns the element present in majority of the list, or -1 otherwise
    """
    counts = [0] * TOTAL_MOVEMENTS
    for el in lst:
        counts[int(el)] += 1
    best = 0
    for i in range(1, TOTAL_MOVEMENTS):
        if counts[best] < counts[i]:
            best = i
    majority = (len(lst) + 2) // 2
    if counts[best] < majority:
        return -1
    return best


def mk(directory):
    folders = directory.split(os.path.sep)
    for i in range(len(folders)):
        so_far = str(os.path.sep).join(folders[:i+1])
        try:
            os.mkdir(so_far)
        except:
            pass

def discount_reaction_indeterminacy(labels):
    new_labels = [u for u in labels]
    n = len(labels) - 1
    for i in range(n):
        if i == 0 or labels[i] != labels[i+1] or i == n - 1:
            start = max(0, i - REACTION_TIME_FRAMES)
            end = i
            for j in range(start, end):
                new_labels[j] = -1
            start = i
            end = min(n + 1, i + REACTION_TIME_FRAMES)
            for j in range(start, end):
                new_labels[j] = -1
    return new_labels


def find_frame_labels(fullpath):
    """Returns `is_washing` status and movement codes for each frame
    """
    filename = os.path.basename(fullpath)
    annotators_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Annotations")

    annotations = []

    # Load the supplementary info, if present.
    # This info is not part of the public dataset;
    # it is hand presence information, generated using the Mediapipe hand tracking neural network
    supplementary_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Supplementary")
    supplementary_filename = os.path.join(supplementary_dir, "hands-" + filename + ".txt")
    frames_with_hands = []
    if os.access(supplementary_filename, os.R_OK):
        with open(supplementary_filename, "r") as f:
            for line in f.readlines():
                try:
                    i = int(line)
                except Exception as ex:
                    break
                frames_with_hands.append(i)

    for a in range(1, TOTAL_ANNOTATORS + 1):
        annotator_dir = os.path.join(annotators_dir, "Annotator" + str(a))
        json_filename = os.path.join(annotator_dir, filename.split(".")[0] + ".json")

        if os.access(json_filename, os.R_OK):
            with open(json_filename, "r") as f:
                try:
                  data = json.load(f)
                except:
                  print("failed to load {}".format(json_filename))
                  continue
                a_annotations = [(data['labels'][i]['is_washing'], data['labels'][i]['code']) for i in range(len(data['labels']))]
                annotations.append(a_annotations)

    num_annotators = len(annotations)
    num_frames = len(annotations[0])
    is_washing = []
    codes = []
    for frame_num in range(num_frames):
        frame_annotations = [annotations[a][frame_num] for a in range(num_annotators)]
        frame_is_washing_any = any([frame_annotations[a][0] for a in range(num_annotators)])
        frame_is_washing_all = all([frame_annotations[a][0] for a in range(num_annotators)])
        frame_codes = [frame_annotations[a][1] for a in range(num_annotators)]
        # treat movement 7 as movement 0
        frame_codes = [0 if code == 7 else code for code in frame_codes]

        if frame_is_washing_all:
            frame_is_washing = 2
        elif frame_is_washing_any:
            frame_is_washing = 1
        else:
            frame_is_washing = 0

        is_washing.append(frame_is_washing)
        if frame_is_washing:
            codes.append(majority_vote(frame_codes))
        else:
            codes.append(-1)

    if len(frames_with_hands):
        if len(frames_with_hands) != len(is_washing):
            if len(frames_with_hands) > len(is_washing):
                print("Incorrect dimensions of the supplementary information: {} vs {}".format(len(frames_with_hands), len(is_washing)))
                frames_with_hands = []
            else:
                # pad with zeroes (no hands detected)
                pad_len = len(is_washing) - len(frames_with_hands)
                frames_with_hands += [0] * pad_len

    if False:
        is_washing = discount_reaction_indeterminacy(is_washing)
        codes = discount_reaction_indeterminacy(codes)

    return is_washing, codes, frames_with_hands, num_annotators


def get_frames(folder, testfiles, trainvalfiles):
    """Get and save frames from matching videos
    """
    # to get the summary afterwards
    N_of_videofiles = 0
    N_of_frames_considered = 0
    N_of_frames_washing = 0  # frames labelled as 'is washing'
    N_of_matching_frames = 0 # frames that had a clear majority label
    N_of_saved_movement_classification_frames = 0
    used_files = 0

    print('Processing folder: ' + folder + ' ...')

    for subdir, dirs, files in os.walk(os.path.join(input_folder, folder)):
        for videofile in files:
            if videofile.endswith(".mp4"):
                #print(videofile)
                N_of_videofiles += 1

                fullpath = os.path.join(subdir, videofile)
                used_files +=1
                is_washing, codes, frames_with_hands, num_annotators = find_frame_labels(fullpath)

                vidcap = cv2.VideoCapture(fullpath)
                is_success, image = vidcap.read()
                frame_number = 0

                if videofile in testfiles:
                    traintest = "test-with-locations"
                elif videofile in trainvalfiles:
                    traintest = "trainval-with-locations"
                else:
                    # unknown location
                    continue

                #print(is_washing)

                while is_success:
                    N_of_frames_considered += 1

                    # for frames with multiple annotators and washing on,
                    # potentially save the frame in its repective movement class set
                    if num_annotators > 1 and is_washing[frame_number] == 2:
                        N_of_frames_washing += 1
                        if codes[frame_number] >= 0:
                            N_of_matching_frames += 1

                            # skip some movement 0 frames
                            if (codes[frame_number] != 0 or random.random() < MOVEMENT_0_PROPORTION):
                                subfolder = str(codes[frame_number])
                                filename = 'frame{}_file_{}.jpg'.format(frame_number, os.path.splitext(videofile)[0])
                                # the name of the file storing the frames includes the frame number and the videofile name
                                save_path_and_name = os.path.join(output_folder, traintest, subfolder, filename)
                                cv2.imwrite(save_path_and_name, image)
                                N_of_saved_movement_classification_frames += 1

                    is_success, image = vidcap.read()
                    frame_number += 1
                    if frame_number > MAX_FRAMES:
                        break


    N_of_nonmatching_frames = N_of_frames_washing - N_of_matching_frames
    print('Number of processed videofiles: ', N_of_videofiles)
    print('Number of considered frames: ', N_of_frames_considered)
    print('Number of frames marked as IS WASHING: ', N_of_frames_washing)
    print('Number of frames marked as IS WASHING that did not have a majority label: ', N_of_nonmatching_frames)
    print('Percentage of frames with a majority label: ', 100.0 * N_of_matching_frames / N_of_frames_washing if N_of_frames_washing else 0)
    print('Number of frames saved for movement classification: ', N_of_saved_movement_classification_frames)
    print('')


def main():
    random.seed(0)

    for movement in range(TOTAL_MOVEMENTS - 1):
        mk(os.path.join(output_folder, "test-with-locations", str(movement)))
        mk(os.path.join(output_folder, "trainval-with-locations", str(movement)))

    testfiles = set()
    trainvalfiles = set()
    with open(os.path.join(input_folder, 'statistics-with-locations.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            current = row[0]
            base = os.path.basename(current)
            current_fnm = os.path.splitext(base)[0]
            if row[1] == 'Reanimācija':
                testfiles.add(row[0])
            elif row[1] != 'unknown':
                trainvalfiles.add(row[0])

    list_of_folders = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    for folder in sorted(list_of_folders):
        get_frames(folder, testfiles, trainvalfiles)

# ----------------------------------------------
if __name__ == "__main__":
    main()
