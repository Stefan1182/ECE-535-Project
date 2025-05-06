# client/router.py

import numpy as np
import cv2

prev_frame_mean = None

def should_offload(distance, frame):
    global prev_frame_mean

    # Always offload processing if an object is within 50 cm
    if distance < 50:
        return True

    # Compute mean brightness of current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_mean = np.mean(gray)

    if prev_frame_mean is None:
        prev_frame_mean = current_mean
        return True  # Offload first frame

    brightness_diff = abs(current_mean - prev_frame_mean)
    prev_frame_mean = current_mean

    # Offload if brightness changes significantly
    if brightness_diff > 10:
        return True

    return False
