import cv2
import base64
import requests
import time
import numpy as np
import board
import adafruit_hcsr04
from local_model import predict_local_action
from router import should_offload

# Initialize camera capture at 320×240
cap = cv2.VideoCapture(0)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
else:
    print("[ERROR] Cannot open camera")

# Set up ultrasonic distance sensor
sensor = adafruit_hcsr04.HCSR04(trigger_pin=board.P8_7, echo_pin=board.P8_9)
SERVER_URL = 'http://192.168.0.76:5000/predict'

def read_distance():
    try:
        return sensor.distance  # Measure distance from ultrasonic sensor
    except RuntimeError:
        return 999  # Return a large distance if reading fails

def get_frame():
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            return frame  # Return captured frame
    # Return a blank image if capture fails
    return np.zeros((240, 320, 3), dtype=np.uint8)

# Map numeric action codes to human-readable labels
action_map = {0: "Left", 1: "Straight", 2: "Right", 3: "Stop"}

while True:
    frame = get_frame()
    distance = read_distance()

    if should_offload(distance, frame):
        # Compress frame and encode to base64 for cloud inference
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        try:
            # Send image and distance to prediction server
            response = requests.post(SERVER_URL, json={'image': img_base64, 'distance': distance}, timeout=2.0)
            action = response.json()['action']
            source = "Cloud"
        except Exception as e:
            print(f"[ERROR] Cloud request failed: {e}")
            # Fallback to local prediction on failure
            action = predict_local_action(frame, distance)
            source = "Local (Fallback)"
    else:
        # Perform local prediction when offloading is not needed
        action = predict_local_action(frame, distance)
        source = "Local"

    # Log action and distance
    print(f"[{source}] Distance: {distance:.1f} cm | Action: {action_map.get(action, 'Unknown')}")
    time.sleep(0.05)

cap.release()
