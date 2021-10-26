#!/usr/bin/env python3

import os
import cv2 as cv
import numpy as np

input_dir = "kaggle-dataset-6classes-minimal"
output_dir = "kaggle-dataset-6classes-of"

N_CLASSES = 7
classes = [str(i) in i in range(N_CLASSES)]

def mk(filename):
    try:
        os.mkdir(filename)
    except Exception as ex:
        pass

# The following frees up resources and
# closes all windows
#cap.release()
#cv.destroyAllWindows()

def extract_flow(c, filename):
    in_fullname = os.path.join(input_dir, c, filename)
    cap = cv.VideoCapture(in_fullname)
    frame_num = 0

    out_fullname = os.path.join(output_dir, c, "frame_{}_" + filename + ".jpg")

    ret, frame = cap.read()

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while ret:
	
	# Converts each frame to grayscale - we previously
	# only converted the first frame to grayscale
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	# Calculates dense optical flow by Farneback method
	flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
					   None,
					   0.5, 3, 15, 3, 5, 1.2, 0)
	
	# Computes the magnitude and angle of the 2D vectors
	magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
	
	# Sets image hue according to the optical flow
	# direction
	mask[..., 0] = angle * 180 / np.pi / 2
	
	# Sets image value according to the optical flow
	# magnitude (normalized)
	mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
	
	# Converts HSV to RGB (BGR) color representation
	rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
	
	# Opens a new window and displays the output frame
	#cv.imshow("dense optical flow", rgb)
	
	# Updates previous frame
	prev_gray = gray

        save_path_and_name = out_fullname.format(frame_num)
        frame_num += 1
        cv2.imwrite(save_path_and_name, rbg)
        print("saved", save_path_and_name)

        # read next frame
	ret, frame = cap.read()

    cap.release()


def main():
    mk(output_dir)
    for c in classes:
        mk(os.path.join(output_dir, c))

    for c in classes:        
        for filename in os.listdir(os.path.join(input_dir, c)):
            if filename[-4:] == ".mp4":
                extract_flow(c, filename)
        for filename in os.listdir(dirname):
        fullname = os.path.join(dirname, filename)


if __name__ == "__main__":
    main()
