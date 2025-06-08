from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_newreward(image) -> float:
    # Helper functions to calculate different luma conditions based on quantiles

    image = np.array(image)  # Convert PIL Image to NumPy array
    image = (image / 255.0).astype(np.float32)

    def average(image):
        return np.mean(image)

    def quantile_mean(image, lower_quantile, upper_quantile):
        lower_bound = np.quantile(image, lower_quantile)
        upper_bound = np.quantile(image, upper_quantile)
        quantile_range = image[(image >= lower_bound) & (image <= upper_bound)]
        return np.mean(quantile_range)

    epsilon = 5 / 255  # Adjust epsilon as needed

    # Calculate rewards based on conditions
    r1 = 1 if (50 / 255 - epsilon) <= average(image) <= (50 / 255 + epsilon) else 0
    # r2 = 1 if (5 / 255 <= quantile_mean(image, 0, 0.2) <= 100 / 255) else 0
    r3 = 1 if (10 / 255 <= quantile_mean(image, 0.2, 0.4) <= 100 / 255) else 0
    r4 = 1 if (12.5 / 255 <= quantile_mean(image, 0.4, 0.6) <= 80 / 255) else 0
    # r5 = 1 if (80 / 255 <= quantile_mean(image, 0.6, 0.8) <= 100 / 255) else 0
    r6 = 1 if (100 / 255 <= quantile_mean(image, 0.9, 1.0) <= 250 / 255) else 0

    # weight1 = 0.1
    # weight2 = 0.3
    # weight3 = 0.3
    # weight4 = 0.1
    # weight5 = 0.01
    # weight6 = 0.3
    weight1 = 0.1
    # weight2 = 0.3
    weight3 = 0.3
    weight4 = 0.1
    # weight5 = 0.01
    weight6 = 0.3
    # Calculate total reward

    total_reward = (r1 * weight1) + (r3 * weight3) + (r4 * weight4) + (r6 * weight6)
    return total_reward

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)  # or yolov5n - yolov5x6, custom
orb = cv2.ORB_create(nfeatures=1500)

# Output directory
for index in range(1,41):
    output_dir = f"camera_angle1\\output_scene{scene}"
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    Num_object = []
    Sum_confidence_score = []
    Stat_reward = []
    Feature_point = []

    # Directory containing images
    image_dir = rf"C:\\Users\\xingang\\Desktop\\AEC_dataset\\ExpoSweep\\training\\moderate\\moderate_scene{index}"
    # Process each image
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)  # Load image
            rotated_img = img.rotate(270, expand=True)
            # Inference
            results = model(rotated_img)
            df = results.pandas().xyxy[0]  # Results as a DataFrame
            Num_object.append(df.shape[0])  # Number of detected objects
            # Check for detections and extract confidence scores
            if df.shape[0] == 0:
                confidence_score = 0
            else:
                confidence_score = df['confidence'].sum()  # You can choose max, mean, etc.

            Sum_confidence_score.append(confidence_score)  # Sum of confidence scores

            stat_reward = calculate_newreward(img)
            Stat_reward.append(stat_reward)  # Stat reward
            # Assuming 'img' is a PIL Image
            img_np = np.array(img)  # Convert PIL Image to NumPy array
            if img_np.ndim == 3:  # Check if the image is colored (3 channels)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            # Now you can use img_np with ORB
            keypoints = orb.detect(img_np, None)
            keypoints, descriptors = orb.compute(img_np, keypoints)
            Feature_point.append(len(keypoints))  # Number of feature points

        # Plotting 
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 3 rows, 1 column
        # Plot Number of Objects Detected
        axs[0, 0].plot(Num_object, marker='o', linestyle='-')
        axs[0, 0].set_title('Number of Detected Objects')
        axs[0, 0].set_xlabel('Image Index')
        axs[0, 0].set_ylabel('Count')
        # Plot Sum of Confidence Scores
        axs[0, 1].plot(Sum_confidence_score, marker='o', linestyle='-')
        axs[0, 1].set_title('Sum of Confidence Scores')
        axs[0, 1].set_xlabel('Image Index')
        axs[0, 1].set_ylabel('Sum Confidence Score')
        # Plot Statistical Reward
        axs[1, 0].plot(Stat_reward, marker='o', linestyle='-')
        axs[1, 0].set_title('Statistical Reward')
        axs[1, 0].set_xlabel('Image Index')
        axs[1, 0].set_ylabel('Reward')
        # Plot Feature Points (assuming Feature_points is defined)
        axs[1, 1].plot(Feature_point, marker='o', linestyle='-')
        axs[1, 1].set_title('Feature Points')
        axs[1, 1].set_xlabel('Image Index')
        axs[1, 1].set_ylabel('Count')
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plot_output_dir = "plot_output"
        os.makedirs(plot_output_dir, exist_ok=True)  # Create the plot output directory if it doesn't exist
        plt.savefig(os.path.join(plot_output_dir, f'reward_analysis_secen{scene}_angle{angle}.png'))
