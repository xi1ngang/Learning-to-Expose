import copy
import io
import math

import random

import cv2

import numpy as np
import torch
# from manifold.clients.python import ManifoldClient
from PIL import Image


def calculate_newreward(image) -> float:
    # Helper functions to calculate different luma conditions based on quantiles

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

def load_dataset(manifold_path, i, j):
    # Initialize the client (authentication details might be needed)

    bucket, path = manifold_path.split("/", 1)
    client = ManifoldClient(bucket)
    # Fetch the list of files in the specified folder

    file_list = [f[0] for f in client.sync_ls(path)]
    # Load each image file into a numpy array and extract ISO and ExpT values

    dataset = []
    for file_name in file_list:
        # Extract ISO and ExpT values from the file name

        try:
            # Generalize the extraction of the idx part

            idx_part = file_name.split(f"205-{j}-")[1].split("-")[0]
            idx_value = int(idx_part)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract idx from file name: {file_name}")
            print(f"Error: {e}")
            continue
        # Load the image

        file_path = f"{path}/{file_name}"
        stream = io.BytesIO()
        client.sync_get(file_path, stream)
        image_data = Image.open(stream)
        image_array = np.array(image_data)
        # Append the image array, ISO value, and exposure time as a tuple

        dataset.append((image_array, idx_value))
    image_data = [{"image": element[0], "idx": element[1]} for element in dataset]
    return image_data


def image_augumentation(image, augmentations):
    crop_height = augmentations[0]
    crop_width = augmentations[1]
    start_row = augmentations[2]
    start_col = augmentations[3]
    # Crop the image
    image_aug = image[
        start_row : start_row + crop_height, start_col : start_col + crop_width
    ]

    # Horizontal flip using slicing
    horizontal = augmentations[2]
    if horizontal:
        image_aug = image_aug[:, ::-1]
    # Vertical flip using slicing
    vertical = augmentations[3]
    if vertical:
        image_aug = image_aug[::-1, :]
    return image_aug


def dataset_aug(dataset, crop_size):
    dataset_augmentation = copy.deepcopy(dataset)
    crop_height = crop_size
    crop_width = crop_size
    black_margin_size = 100
    image = dataset[0]["image"]
    max_start_row = image.shape[0] - crop_height - black_margin_size
    max_start_col = image.shape[1] - crop_width - black_margin_size
    start_row = np.random.randint(
        black_margin_size, max_start_row + 1
    )  # +1 because the upper bound is exclusive
    start_col = np.random.randint(black_margin_size, max_start_col + 1)
    is_flip_horizontal = np.random.choice([True, False])
    is_flip_vertical = np.random.choice([True, False])
    augmentations = [
        crop_height,
        crop_width,
        start_row,
        start_col,
        is_flip_horizontal,
        is_flip_vertical,
    ]
    for i in range(len(dataset)):
        image = dataset[i]["image"]
        image_aug = image_augumentation(image, augmentations)
        dataset_augmentation[i]["image"] = image_aug
    return dataset_augmentation


def feature_extractor(image_tensor):

    image = image_tensor.squeeze()
    mean_brightness = torch.mean(image)
    column_sum = torch.sum(image, dim=0) / image.shape[0]
    return column_sum, mean_brightness


def find_optimal_idx(image_data, target_brightness):

    min_difference = float("inf")
    optimal_idx = -1
    for image, idx in zip(image_data["image"], image_data["idx"]):
        mean_brightness = image.mean()
        diff = abs(mean_brightness - target_brightness)
        if diff < min_difference:
            min_difference = diff
            optimal_idx = idx.item()
    return optimal_idx


def select_action(
    state, model, action_space, EPS_END, EPS_START, EPS_DECAY, steps_done, device
):
    # pyre-ignore
    # Always select a random action for the first 10,000 steps

    if steps_done < 50000:
        steps_done += 1
        return (
            torch.tensor([[action_space.sample()]], device=device, dtype=torch.long),
            steps_done,
        )
    sample = random.random()
    # sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        # print("Using the policy network")

        with torch.no_grad():
            return model(state).max(1).indices.view(1, 1), steps_done
    else:
        # print("Using random action")
        # Use the sample method for exploration

        return (
            torch.tensor([[action_space.sample()]], device=device, dtype=torch.long),
            steps_done,
        )
