import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def calculate_reward(image) -> float:
    # Helper functions to calculate different luma conditions based on quantiles

    def average(image: torch.Tensor) -> torch.Tensor:
        return torch.mean(image)

    def quantile_mean(
        image: torch.Tensor, lower_quantile: float, upper_quantile: float
    ) -> torch.Tensor:
        lower_bound = torch.quantile(image, lower_quantile)
        upper_bound = torch.quantile(image, upper_quantile)
        quantile_range = image[(image >= lower_bound) & (image <= upper_bound)]
        return torch.mean(quantile_range)

    image = image.float() / 255.0

    epsilon = 5 / 255  # Adjust epsilon as needed
    # Calculate rewards based on conditions
    r1 = 1 if (50 / 255 - epsilon) <= average(image) <= (50 / 255 + epsilon) else 0
    r2 = 1 if (5 / 255 <= quantile_mean(image, 0, 0.2) <= 100 / 255) else 0
    r3 = 1 if (10 / 255 <= quantile_mean(image, 0.2, 0.4) <= 100 / 255) else 0
    r4 = 1 if (12.5 / 255 <= quantile_mean(image, 0.4, 0.8) <= 80 / 255) else 0
    r5 = 1 if (100 / 255 <= quantile_mean(image, 0.8, 1.0) <= 250 / 255) else 0

    weight1 = 0.1
    weight2 = 0.3
    weight3 = 0.3
    weight4 = 0.1
    weight5 = 0.3
    # Calculate total reward

    reward = (
        (r1 * weight1)
        + (r2 * weight2)
        + (r3 * weight3)
        + (r4 * weight4)
        + (r5 * weight5)
    )
    return reward

# exposure_index = [20, 10, 7, 7]
reward = []
scene_index = 9

for index in range(1,112):
    image_path = f"C:\\Users\\xingang\\Desktop\\AEC_dataset\\ExpoSweep\\training\\neutral\\neutral_scene{scene_index}\\{index}.png"
    image = Image.open(image_path)
    image_array = np.array(image)  # Convert to a NumPy array
    image_tensor = torch.tensor(image_array)
    reward.append(calculate_reward(image_tensor))
# image_path = f"C:\\Users\\xingang\\Desktop\\AEC_dataset\\ExpoSweep\\evaluation\\outdoor\\outdoor_scene1\\4.png"
# image = Image.open(image_path)
# image = np.array(image)

# print(reward)

# # Create the plot
# plt.figure(figsize=(10, 5))
# plt.plot([1,2,3,4], exposure_index, color='blue', linewidth=2.5, marker='o', markersize=16)

# # Add labels and title
# plt.xlabel('Inference Step', fontsize=28)
# plt.ylabel('Exposure Index', fontsize=28)

# # Set font size for tick labels
# plt.xticks(fontsize=28)
# plt.yticks(fontsize=28)
# # Remove the upper and right spines
# ax = plt.gca()  # Get the current axes
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# Adjust layout and save the plot
# plt.tight_layout()
# plt.savefig('./eval_plot1.png')




# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(range(1,112), reward, color='blue', linewidth=2.5)

# Add labels and title
plt.xlabel('Exposure Index', fontsize=18)
plt.ylabel('Reward', fontsize=18)

# Set font size for tick labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
# Remove the upper and right spines
ax = plt.gca()  # Get the current axes
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('./eval_plot2.png')
