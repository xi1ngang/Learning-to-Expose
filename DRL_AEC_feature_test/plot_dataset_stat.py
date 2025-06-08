import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)






def categorize_brightness(json_data):
    brightness_counts = {'dark': 0, 'moderate': 0, 'bright': 0}
    for key, value in json_data.items():
        average_luma = value['average_luma']
        if average_luma < 50:
            category = 'dark'
        elif 50 <= average_luma <= 80:
            category = 'moderate'
        else:
            category = 'bright'
        
        # Add the brightness category to the JSON data
        value['brightness_category'] = category
        
        # Increment the count for the category
        brightness_counts[category] += 1
    return brightness_counts
# Main function to process files and plot data


def main():
    # List of JSON files
    json_files1 = ['.\\eval_indoor_scene_analysis.json', '.\\eval_outdoor_scene_analysis.json']  # Replace with your actual file paths
    json_files2 = ['.\\train_bright_scene_analysis.json', '.\\train_dim_scene_analysis.json', '.\\train_neutral_scene_analysis.json']  # Replace with your actual file paths
    json_eval_indoor = ['.\\eval_indoor_scene_analysis.json']  # Replace with your actual file paths
    json_eval_outdoor = ['.\\eval_outdoor_scene_analysis.json']  # Replace with your actual file paths
    json_train_dim = ['.\\train_dim_scene_analysis.json']  # Replace with your actual file paths
    json_train_bright = ['.\\train_bright_scene_analysis.json']  # Replace with your actual file paths
    json_train_neutral = ['.\\train_neutral_scene_analysis.json']  # Replace with your actual file paths
    # Initialize lists to store combined data
    average_luma1 = []
    dynamic_range_ratio1 = []
    average_luma2 = []
    dynamic_range_ratio2 = []
    average_luma_dim = []
    dynamic_range_ratio_dim = []
    average_luma_bright = []
    dynamic_range_ratio_bright = []
    average_luma_neutral = []
    dynamic_range_ratio_neutral = []
    average_luma_indoor = []
    dynamic_range_ratio_indoor = []
    average_luma_outdoor = []
    dynamic_range_ratio_outdoor = []
    # Process each file
    for file in json_files1:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma1.append(value['average_luma'])
                dynamic_range_ratio1.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")
    for file in json_files2:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma2.append(value['average_luma'])
                dynamic_range_ratio2.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")

    for file in json_train_bright:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma_bright.append(value['average_luma'])
                dynamic_range_ratio_bright.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")

    for file in json_train_dim:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma_dim.append(value['average_luma'])
                dynamic_range_ratio_dim.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")

    for file in json_train_neutral:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma_neutral.append(value['average_luma'])
                dynamic_range_ratio_neutral.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")

    for file in json_eval_indoor:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma_indoor.append(value['average_luma'])
                dynamic_range_ratio_indoor.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")

    for file in json_eval_outdoor:
        if os.path.exists(file):
            data = load_json(file)
            for key, value in data.items():
                average_luma_outdoor.append(value['average_luma'])
                dynamic_range_ratio_outdoor.append(value['dynamic_range_ratio'])
        else:
            print(f"File {file} not found.")
    print(np.min(average_luma_indoor), np.mean(average_luma_indoor), np.max(average_luma_indoor))
    print(np.min(dynamic_range_ratio_indoor), np.mean(dynamic_range_ratio_indoor), np.max(dynamic_range_ratio_indoor) )
    print(np.min(average_luma_outdoor), np.mean(average_luma_outdoor), np.max(average_luma_outdoor) )
    print(np.min(dynamic_range_ratio_outdoor), np.mean(dynamic_range_ratio_outdoor), np.max(dynamic_range_ratio_outdoor) )
    # Plot the combined data
    # plot_data(all_average_luma, all_dynamic_range_ratio, all_low_exposure_saturated_percentage, all_high_exposure_dark_percentage)
        # Plot histograms
    # plt.rc('font', size=20)  # You can adjust the size as needed
    # plt.figure(figsize=(10, 8))
    # plt.scatter(average_luma2, dynamic_range_ratio2, alpha=0.7, color='b', label='Training Scenes', s=80, marker='o')
    # plt.scatter(average_luma_indoor, dynamic_range_ratio_indoor, alpha=0.7, color='r', label='Indoor Evaluation Scenes', s=150, marker='^')
    # plt.scatter(average_luma_outdoor, dynamic_range_ratio_outdoor, alpha=0.7, color='g', label='Outdoor Evaluation Scenes', s=150, marker='^')
    # plt.xlabel('Average Intensity', fontsize=20)
    # plt.ylabel('Dynamic Range Ratio', fontsize=20)
    # plt.legend()
    # # plt.grid(True)
    
    plt.rc('font', size=16)  # You can adjust the size as needed
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(average_luma2 + average_luma1, bins=15, kde=True, color='b')
    # sns.histplot(average_luma1, bins=15, kde=True, color='r')
    plt.xlabel('Average Brightness', fontsize=16)
    plt.ylabel('Count', fontsize=16)

    plt.subplot(1, 2, 2)
    sns.histplot(dynamic_range_ratio2 + dynamic_range_ratio1, bins=15, kde=True, color='g')
    # sns.histplot(dynamic_range_ratio1, bins=15, kde=True, color='r')
    plt.xlabel('Dynamic Range Ratio', fontsize=16)
    plt.ylabel('Count', fontsize=16)

    plt.tight_layout()
    plt.savefig('./datasetstat_hist_plots.png')
    plt.savefig('./datasetstat_hist_plots.pdf', format='pdf')


    # print(f"Number of each category: {total_brightness_counts}")
if __name__ == "__main__":
    main()
