# server/mpc_controller.py

def compute_control(action):
    """Compute simple control commands based on predicted action.

    Args:
        action (int): 0 = Left, 1 = Straight, 2 = Right, 3 = Stop

    Returns:
        steering (float): steering angle (-1 to 1, left to right)
        velocity (float): target speed (m/s)
    """
    if action == 0:  # Left turn
        steering = -0.5  # Moderate left steering
        velocity = 1.5
    elif action == 1:  # Straight
        steering = 0.0   # No steering change
        velocity = 2.0
    elif action == 2:  # Right turn
        steering = 0.5   # Moderate right steering
        velocity = 1.5
    elif action == 3:  # Stop
        steering = 0.0
        velocity = 0.0
    else:
        steering = 0.0
        velocity = 0.0

    return steering, velocity
