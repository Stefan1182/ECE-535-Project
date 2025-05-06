# server/server.py

from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from mpc_controller import compute_control
from cloud_model import predict_action_and_overlay

app = Flask(__name__)

last_frame = None
last_action = None
last_control = (0.0, 0.0)  # (steering, velocity)
action_map = {0: "Left", 1: "Straight", 2: "Right", 3: "Stop"}

@app.route('/predict', methods=['POST'])
def predict():
    global last_frame, last_action, last_control
    data = request.json

    try:
        # Decode base64 image and convert to OpenCV format
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Image decode failed: {e}")
        return jsonify({'error': 'Invalid image'})

    try:
        distance = float(data.get('distance', 999))
    except:
        distance = 999

    # Predict action and get annotated frame from cloud model
    last_action, vis_frame = predict_action_and_overlay(img, distance)
    last_frame = vis_frame.copy()

    # Compute control commands based on predicted action
    last_control = compute_control(last_action)

    return jsonify({
        'action': last_action,
        'steering': last_control[0],
        'velocity': last_control[1]
    })

def display_frames():
    global last_frame, last_action, last_control

    while True:
        if last_frame is not None:
            frame = last_frame.copy()
            action_text = action_map.get(last_action, "Unknown")
            steering, velocity = last_control

            # Overlay action and control info on frame
            cv2.putText(frame, f"Action: {action_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Steering: {steering:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Velocity: {velocity:.2f} m/s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Server Live View", frame)

        # Quit display loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    print("✅ server.py is starting Flask + display thread ...")
    import threading
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    ).start()

    display_frames()
