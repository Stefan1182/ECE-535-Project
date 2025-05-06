import torch
import numpy as np
import cv2
from lane_model import detect_lanes

# Automatically select device (use GPU for inference if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5n model (autoshape enabled by default, handles inference preprocessing + NMS)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device).eval().half()
CLASS_NAMES = model.names

# Print debug information
print("[DEBUG] CUDA available:", torch.cuda.is_available())
print("[DEBUG] Model device:", next(model.parameters()).device)
print("[CHECK] YOLOv5 model device:", next(model.parameters()).device)

def predict_action_and_overlay(img, distance):
    img_vis = img.copy()

    # Pass numpy image directly to Ultralytics (automatically handles RGB conversion and preprocessing)
    with torch.no_grad():
        results = model(img_vis[..., ::-1])  # Convert BGR to RGB for model

    # Extract predictions after automatic NMS: [x1, y1, x2, y2, confidence, class]
    preds = results.pred[0].cpu().numpy()

    center_blocked = False
    left_clear = True
    right_clear = True

    for det in preds:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        label = CLASS_NAMES[cls]
        cx = (x1 + x2) / 2  # Compute center x-coordinate of detection

        if cls in [0, 2]:  # If detection is person or car
            if 0.4 * img.shape[1] < cx < 0.6 * img.shape[1]:
                center_blocked = True
            elif cx <= 0.4 * img.shape[1]:
                left_clear = False
            elif cx >= 0.6 * img.shape[1]:
                right_clear = False

        # Draw bounding box and label on visualization image
        cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img_vis, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Perform lane detection using LaneNet
    lanes = detect_lanes(img_vis)
    if len(lanes) == 0:
        cv2.putText(img_vis, "No lanes detected", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        for lane in lanes:
            for i in range(1, len(lane)):
                cv2.line(img_vis, lane[i - 1], lane[i], (0, 255, 255), 2)

    # Decision logic based on distance and obstacle/lane information
    if distance < 20:
        action = 3  # Stop if too close
    elif center_blocked:
        if left_clear and len(lanes) >= 2:
            action = 0  # Turn left if center blocked and left lane clear
        elif right_clear and len(lanes) >= 2:
            action = 2  # Turn right if center blocked and right lane clear
        else:
            action = 3  # Stop if no clear lane available
    else:
        action = 1  # Go straight otherwise

    return action, img_vis
