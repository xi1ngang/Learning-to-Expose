import os
from PIL import Image
import numpy as np
import random

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

def adjust_exposure(image_path, initial_exposure):
    # PID controller parameters
    kp = 0.1
    ki = 0.1
    kd = 0.05
    setpoint = 80
    min_exposure = 1
    max_exposure = 111

    # Initialize PID controller
    pid = PIDController(kp, ki, kd, setpoint, min_exposure, max_exposure)
    
    # Start with the initial exposure
    current_exposure = initial_exposure
    running_index = []
    
    for iteration in range(100):  # Limit iterations to prevent infinite loop

        running_index.append(current_exposure)

        image_file = os.path.join(image_path, f"{current_exposure}.png")
        with Image.open(image_file) as img:
        # Simulate exposure adjustment loop
            # Calculate current brightness
            current_brightness = calculate_brightness(img)
            # Update exposure using PID controller
            new_exposure = pid.update(current_brightness)
            print(f"Iteration: {iteration}, Current Brightness: {current_brightness:.2f}, New Exposure: {new_exposure}")
            # Check for convergence (e.g., if the brightness is close enough to the setpoint)
            if abs(current_brightness - setpoint) < 0.5:
                print("Converged to target brightness. Convergence speed: ", len(running_index)-1, "iterations and moving index is", running_index)
                break
            
            # Simulate applying the new exposure (this is conceptual)
            # In a real scenario, you would adjust the camera settings and capture a new image
            current_exposure = new_exposure

# Path to the images
image_path = r"C:\Users\xingang\Desktop\AEC_dataset\ExpoSweep\evaluation\indoor\bright_scene1"

# Initial exposure index
initial_exposure = random.randint(1, 111)

# Adjust exposure for the image
adjust_exposure(image_path, initial_exposure)
