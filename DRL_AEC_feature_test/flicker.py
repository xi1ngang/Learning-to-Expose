import numpy as np
import torch

def is_converge(index_array):

    differences = np.diff(index_array)
    print(differences)
    differences2examine = differences[-5:]
    print(differences2examine)
    return all(abs(value) <= 2 for value in differences2examine)

    # normalized_differences = differences / np.mean(np.abs(index_array))

    # direction_changes = np.sign(normalized_differences[:-1]) != np.sign(normalized_differences[1:])
    # flicker_metric = np.sum(np.abs(normalized_differences[1:][direction_changes]))
    # flicker_metric += np.std(normalized_differences)

    # return flicker_metric


# Example arrays
x = [5, 4, 3, 2, 1, 1, 1, 1, 1, 1]  # Expected low nonsmoothness
y = [5, 4, 3, 2, 1, 0, 1, 2, 1, 0]  # Expected high nonsmoothness
z = [110, 50, 30, 20, 20, 20, 20, 20, 20, 20]  # Expected low nonsmoothness
k = [200, 50, 30, 20, 20, 20, 20, 21, 20, 22, 20, 22, 20]  # Expected low nonsmoothness

# Calculate nonsmoothness metric
# print("Array x nonsmoothness metric:", flicker_analysis(x))
# print("Array y nonsmoothness metric:", flicker_analysis(y))
# print("Array z nonsmoothness metric:", flicker_analysis(z))
# print("Stability of k:", is_converge(k))

# Define the scenarios and their corresponding difficulties
scenarios = ["moderate", "bright", "dark"]
difficulties = [1, 2, 3]

# Initialize the scene index for each scenario
scene_indices = {scenario: 0 for scenario in scenarios}

# Define the number of scenes for each scenario
num_scenes_per_scenario = {
    "moderate": 40,
    "bright": 31,
    "dark": 29
}

num_moderate = 0
num_bright = 0
num_dark = 0

# Main loop
num_epochs = 1500
scene_index = 0
for num_epoch in range(num_epochs):
    # Load the dataset every 10 epochs
    if num_epoch % (num_epochs / 100) == 0:
        # Get the current scenario based on the difficulty level
        if num_moderate < 40:
            current_scenario = "moderate"
            num_moderate += 1
            scene_index = num_moderate
        elif num_bright < 31:
            current_scenario = "bright"
            num_bright += 1
            scene_index = num_bright
        elif num_dark < 29:
            current_scenario = "dark"
            num_dark += 1
            scene_index = num_dark
        print(f"Loading dataset for scenario {current_scenario} scene {scene_index}")
