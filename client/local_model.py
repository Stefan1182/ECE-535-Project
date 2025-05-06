# client/local_model.py

import numpy as np
import cv2

def predict_local_action(frame, distance):
    # Emergency stop if the obstacle is too close
    if distance < 20:
        return 3  # Stop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Split image into left and right halves
    left = gray[:, :w // 2]
    right = gray[:, w // 2:]

    left_mean = np.mean(left)
    right_mean = np.mean(right)

    # Decide direction based on brightness
    if left_mean < right_mean - 10:
        return 2  # Turn right
    elif right_mean < left_mean - 10:
        return 0  # Turn left
    else:
        return 1  # Go straight
