# server/lane_model.py

import torch
import cv2
import numpy as np
import os
import sys

# Add Ultra-Fast-Lane-Detection path to module search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Ultra-Fast-Lane-Detection'))

from model.model import parsingNet
from torchvision import transforms

# Model configuration: grid dimensions and backbone type
cls_dim = (101, 56, 4)
backbone = '18'

# Automatically select device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running lane detection on: {device}")

# Initialize parsingNet model with FP16 precision and move to device
net = parsingNet(pretrained=False, backbone=backbone, cls_dim=cls_dim).to(device).half()
state_dict = torch.load('checkpoints/tusimple_18.pth', map_location=device)
net.load_state_dict(state_dict['model'], strict=False)
net.eval()

# Image preprocessing pipeline (resize, normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def detect_lanes(img):
    """
    Takes a BGR image as input and returns a list of valid lane point sequences.

    Model output shape is (101, 56, 4):
    - 101: number of horizontal grid partitions
    - 56: number of vertical anchors
    - 4: maximum number of lanes
    """
    orig_h, orig_w = img.shape[:2]
    # Resize image for model input
    img_resize = cv2.resize(img, (800, 288))
    img_tensor = transform(img_resize).unsqueeze(0).to(device).half()  # Convert input to half precision

    with torch.no_grad():
        out = net(img_tensor)         # Raw model output tuple
        out = out[0].cpu().numpy()    # Convert to numpy array with shape (101, 56, 4)
        print(f"[DEBUG] LaneNet out: shape={out.shape}, max={out.max():.2f}, min={out.min():.2f}")

    # Rearrange to shape (101, 4, 56)
    out = out.transpose(0, 2, 1)
    # Choose the grid cell with highest confidence for each (lane, anchor)
    out = out.argmax(axis=0)         # Now shape = (4, 56)

    lanes = []
    griding_num = 100  # Number of horizontal grid divisions (0~100)
    # Iterate over each potential lane
    for i in range(out.shape[0]):  # Loop through 4 lanes
        lane = []
        # Iterate over vertical anchors
        for j in range(out.shape[1]):  # 56 vertical anchors
            col = out[i, j]
            if col != griding_num:  # Ignore grid positions with no detection
                x = int(col * orig_w / griding_num)
                y = int(j * orig_h / out.shape[1])
                lane.append((x, y))

        # Mild filtering to remove spurious lane lines
        xs = [pt[0] for pt in lane]
        if len(xs) >= 3 and (max(xs) - min(xs)) > 10:
            lanes.append(lane)

    return lanes
