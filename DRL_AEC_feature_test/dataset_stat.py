import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
import numpy as np
import json

def analyze_scene(image_dir, expo_idx, target_luma):
    """
    Analyze an image based on average luma, dynamic range, and staggered dynamic range.
    Args:
        image_path (str): Path to the image file.
        exposure_index_low (int): The lowest exposure index for staggered dynamic range.
        exposure_index_high (int): The highest exposure index for staggered dynamic range.
    Returns:
        dict: A dictionary containing the average luma, dynamic range, and staggered dynamic range.
    """
    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Find the target image based on the filename pattern
    difference = []
    sorted_list = sorted(image_files, key=lambda x: int(x.split('.')[0]))
    for image_file in sorted_list:
        # Extract the exposure index from the filename
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        image = np.array(image)
        # Calculate average luma
        average_luma = np.mean(image)
        difference.append(abs(average_luma - target_luma))
    opt_index = int(np.argmin(difference) + 1)
    # Read the image to calculate dynamic range
    image4dr = Image.open(os.path.join(image_dir, f'{opt_index}.png'))
    # Convert PIL Image to NumPy array
    image4dr = np.array(image4dr)
    # Calculate dynamic range
    P1 = np.percentile(image4dr, 1)
    P99 = np.percentile(image4dr, 99)

    # Compute the linear dynamic range
    linear_dynamic_range = P99 - P1

    # Compute the logarithmic dynamic range in dB
    dynamic_range_ratio = 20 * np.log10((P99) / (P1 + 1e-6))

    # Calculate average luma
    image4avg = Image.open(os.path.join(image_dir, f'{expo_idx}.png'))
    image4avg = np.array(image4avg)
    average_luma = np.mean(image4avg)
    
    return {
        'image_path': f"{image_dir}",
        'average_luma': average_luma,
        'dynamic_range_ratio': dynamic_range_ratio
    }
# Example usage




results = {}
expo_idx = 50
target_luma = 80
index = 1
root_dir = f"C:\\Users\\xingang\\Desktop\\AEC_dataset\\ExpoSweep\\evaluation\\indoor"
for subdir_name in os.listdir(root_dir):
    image_dir = os.path.join(root_dir, subdir_name)
    result = analyze_scene(image_dir, expo_idx, target_luma)
    if result:
        results[f"scene{index}"] = result
        index += 1

        

    # Save results to a JSON file
    with open(f'eval_indoor_scene_analysis.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
