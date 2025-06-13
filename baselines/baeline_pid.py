import os
from PIL import Image
import numpy as np
import random
import torch

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, min_exposure, max_exposure):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.integral = 0
        self.previous_error = 0

    def update(self, current_brightness):
        # Calculate error
        error = self.setpoint - current_brightness
        
        # Proportional term
        p = self.kp * error
        
        # Integral term
        self.integral += error
        i = self.ki * self.integral
        
        # Derivative term
        derivative = error - self.previous_error
        d = self.kd * derivative
        
        # PID output
        output = p + i + d
        
        # Update previous error
        self.previous_error = error
        
        # Clamp the output to the exposure index range and convert to integer
        new_exposure = int(max(self.min_exposure, min(self.max_exposure, output)))
        
        return new_exposure

def calculate_brightness(image):
    image = np.array(image)
    # Calculate average luma
    average_luma = np.mean(image)
    return average_luma


def calculate_reward(image: np.ndarray) -> float:
    """
    Calculate a reward based on image brightness distribution using NumPy.

    Args:
        image (np.ndarray): Input image as a NumPy array with pixel values in [0, 255].

    Returns:
        float: Reward value.
    """

    # Normalize image to [0, 1]
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    flat_image = image.flatten()

    def average(img: np.ndarray) -> float:
        return np.mean(img)

    def quantile_mean(img: np.ndarray, lower_q: float, upper_q: float) -> float:
        lower_bound = np.quantile(img, lower_q)
        upper_bound = np.quantile(img, upper_q)
        mask = (img >= lower_bound) & (img <= upper_bound)
        return np.mean(img[mask]) if np.any(mask) else 0.0

    epsilon = 5 / 255

    # Conditions
    r1 = 1 if (50/255 - epsilon) <= average(flat_image) <= (50/255 + epsilon) else 0
    r2 = 1 if (5/255 <= quantile_mean(flat_image, 0, 0.2) <= 100/255) else 0
    r3 = 1 if (10/255 <= quantile_mean(flat_image, 0.2, 0.4) <= 100/255) else 0
    r4 = 1 if (12.5/255 <= quantile_mean(flat_image, 0.4, 0.8) <= 80/255) else 0
    r5 = 1 if (100/255 <= quantile_mean(flat_image, 0.8, 1.0) <= 250/255) else 0

    # Weights
    weight1 = 0.1
    weight2 = 0.3
    weight3 = 0.3
    weight4 = 0.1
    weight5 = 0.3

    # Final reward
    stat_reward = (
        (r1 * weight1)
        + (r2 * weight2)
        + (r3 * weight3)
        + (r4 * weight4)
        + (r5 * weight5)
    )

    return stat_reward


def compute_smoothness(actions: list | np.ndarray) -> float:
    """
    Compute the smoothness reward of an action trajectory.

    Args:
        actions: A 1D list or NumPy array of scalar actions (length ≥ 2)

    Returns:
        A float in [0, 1] representing the smoothness reward
    """
    actions = np.array(actions, dtype=np.float32)
    
    if len(actions) < 2:
        return 1.0  # Consider a single-element sequence perfectly smooth

    numerator = abs(actions[0] - actions[-1])
    denominator = np.sum(np.abs(np.diff(actions)))

    return float(numerator / denominator) if denominator != 0 else 1.0


def adjust_exposure(image_path, initial_exposure):
    # PID controller parameters
    kp = 0.1
    ki = 0.1
    kd = 0.05
    setpoint = 60
    min_exposure = 1
    max_exposure = 111

    # Initialize PID controller
    pid = PIDController(kp, ki, kd, setpoint, min_exposure, max_exposure)
    
    # Start with the initial exposure
    current_exposure = initial_exposure
    running_index = []
    
    for iteration in range(50):  # Limit iterations to prevent infinite loop

        running_index.append(current_exposure)

        image_file = os.path.join(image_path, f"{current_exposure}.png")
        with Image.open(image_file) as img:
        # Simulate exposure adjustment loop
            # Calculate current brightness
            current_brightness = calculate_brightness(img)
            # Update exposure using PID controller
            new_exposure = pid.update(current_brightness)
            # print(f"Iteration: {iteration}, Current Brightness: {current_brightness:.2f}, New Exposure: {new_exposure}")
            # Check for convergence (e.g., if the brightness is close enough to the setpoint)
            if abs(current_brightness - setpoint) < 5:
                reward = calculate_reward(img)
                print(f"Reward: {reward}")
                # print("Converged to target brightness. Convergence speed: ", len(running_index)-1, "iterations and moving index is", running_index)
                smoothness = compute_smoothness(running_index)
                print(f"Smoothness: {smoothness}")

                return reward, smoothness, len(running_index)-1
            
            # Simulate applying the new exposure (this is conceptual)
            # In a real scenario, you would adjust the camera settings and capture a new image
            current_exposure = new_exposure

    return 0, 0, 0

# Path to the images
base_path = "/Users/xingang/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Learning-to-Expose/ExpoSweep"




folder_path = os.path.join(base_path, "evaluation", "outdoor")
subfolder_list = os.listdir(folder_path)

total_state_reward = []
total_smoothness = []
total_convergence_speed = []
not_converged = 0

for subfolder in subfolder_list:
    image_path = os.path.join(folder_path, subfolder)
    initial_exposure_list = [4,51,89]

    # Adjust exposure for the image
    for initial_exposure in initial_exposure_list:
        state_reward, smoothness, convergence_speed = adjust_exposure(image_path, initial_exposure)
        if state_reward == 0:
            not_converged += 1
            continue
        total_state_reward.append(state_reward)
        total_smoothness.append(smoothness)
        total_convergence_speed.append(convergence_speed)

print(f"Average state reward: {np.mean(total_state_reward)}")
print(f"Average smoothness: {np.mean(total_smoothness)}")
print(f"Average convergence speed: {np.mean(total_convergence_speed)}")
print(f"Convergence rate: {1-not_converged/(3*len(subfolder_list))}")
print(f"Not converged: {not_converged}")