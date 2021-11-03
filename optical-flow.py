#!/usr/bin/env python3

import os
import cv2 as cv
import numpy as np

#input_dir = "../kaggle-dataset-6classes/videos/test"
#output_dir = "../kaggle-dataset-6classes/of/test"

input_dir = "/data/handwash/RSU_MITC_preprocessed/videos/test"
output_dir = "/data/handwash/RSU_MITC_preprocessed/of/test"


N_CLASSES = 7
classes = [str(i) for i in range(N_CLASSES)]

FPS = 16 if "MITC" in input_dir else 30

# what step to use for movements?
frame_step = FPS // 3

def mk(filename):
    try:
        os.mkdir(filename)
    except Exception as ex:
        pass


def read_frames(video_filename):
    cap = cv.VideoCapture(video_filename)

    frames = []
    ret, frame = cap.read()

    while ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)
        ret, frame = cap.read()

    cap.release()
    return frames


def frame_sequence_to_flow(f1, f2):
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(f1, f2,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros((len(f1), len(f1[0]), 3), dtype=np.float32)

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    return rgb


def extract_flow(c, filename):
    in_fullname = os.path.join(input_dir, c, filename)
    out_fullname = os.path.join(output_dir, c, "frame_{}_" + os.path.splitext(filename)[0] + ".jpg")

    print("processing", in_fullname)
    frames = read_frames(in_fullname)

    n = len(frames)
    flow_frame_num = 0
    for i in range(n - frame_step):
        frame1 = frames[i]
        frame2 = frames[i + frame_step]
        flow_frame = frame_sequence_to_flow(frame1, frame2)

        save_path_and_name = out_fullname.format(flow_frame_num)
        flow_frame_num += 1
        cv.imwrite(save_path_and_name, flow_frame)
        #print("saved", save_path_and_name)


def main():
    mk(output_dir)
    for c in classes:
        mk(os.path.join(output_dir, c))

    for c in classes:
        for filename in os.listdir(os.path.join(input_dir, c)):
            if filename[-4:] == ".mp4":
                extract_flow(c, filename)


if __name__ == "__main__":
    main()
