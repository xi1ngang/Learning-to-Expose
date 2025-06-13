# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import random
from typing import Dict, List, TypedDict

import numpy as np

import torch

from utils import feature_extractor


class ImageData(TypedDict):
    image: torch.Tensor
    idx: int


# pyre-strict
class ActionSpace:
    """
    A class representing the action space for the camera capture environment.
    """

    def __init__(
        self, image_data_aug: List[ImageData], idx_gap: int = 1, N: int = 2
    ) -> None:
        """
        Initializes the action space with the given parameters.
        Args:
            image_data_aug (list[dict]): The augmented image dataset.
            idx_gap (int): The gap between indices in the action space. Defaults to 1.
            N (int): The size of the side window of action space. Defaults to 2.
        """
        self.candidate_idx: list[int] = [
            image_data_aug[i]["idx"] for i in range(len(image_data_aug))
        ]
        self.idx_gap = idx_gap
        self.N = N
        self.idx_actions: list[int] = []

    def generate_action_mapping(self, current_idx: int) -> None:
        """
        Generates the action mapping based on the current index.
        Args:
            current_idx (int): The current index.
        """
        """
        for corner cases, i.e., when current_idx is close to the boundary, we need to adjust the action space accordingly.
        I tried to padding the action space, for example, if current_idx is 0, action space size is 5, then the action space becomes [0 0 0 1 2], 
        which is not ideal based on some preliminary experiemnts, as this action space will
        increase the possibility that 0 to be chosen, and hence the agent will stuck at boundary values and learning becomes unstable
        Now instead of padding, we shift the whole action space if the reuslting one it out of the valid range. For example, if current idx is 0 and action space size is 5, 
        the action space becomes [0 1 2 3 4 5] by using the code below.
        """
        self.idx_actions = []
        for n in range(-self.N, self.N + 1):
            idx = current_idx + n * self.idx_gap
            self.idx_actions.append(idx)
        # Shift the action space to the valid range if necessary
        if self.idx_actions[0] < self.candidate_idx[0]:
            shift_amount = self.candidate_idx[0] - self.idx_actions[0]
            self.idx_actions = [x + shift_amount for x in self.idx_actions]
        if self.idx_actions[-1] > self.candidate_idx[-1]:
            shift_amount = self.idx_actions[-1] - self.candidate_idx[-1]
            self.idx_actions = [x - shift_amount for x in self.idx_actions]
        # Ensure all elements are integers
        self.idx_actions = [int(x) for x in self.idx_actions]

    def get_action(self, current_idx: int, action_index: int) -> int:
        """
        Gets the action at the specified index.
        Args:
            current_idx (int): The current index.
            action_index (int): The index of the action.
        Returns:
            int: The action at the specified index.
        """
        if len(self.idx_actions) == 0:
            self.generate_action_mapping(current_idx)
        return self.idx_actions[action_index]

    def get_action_space_size(self) -> int:
        """
        Gets the size of the action space.
        Returns:
            int: The size of the action space.
        """
        return len(self.idx_actions)

    def sample(self) -> int:
        """
        Samples a random action from the action space.
        Returns:
            int: A random action from the action space.
        """
        return random.randint(0, 2 * self.N - 1)


class CameraCaptureEnv:
    """
    A class representing the camera capture environment.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        image_data_aug: List[ImageData],
        optimal_idx: int = 1,
        N: int = 2,
    ) -> None:
        """
        Initializes the environment with the given parameters.
        Args:
            action_space (ActionSpace): The action space.
            image_data_aug (list[dict]): The augmented image data.
            optimal_idx (int): The optimal index. Defaults to None.
            N (int): The size of the action space. Defaults to None.
        """
        self.candidate_idx: list[int] = [
            image_data_aug[i]["idx"] for i in range(len(image_data_aug))
        ]
        self.current_idx: int = random.choice(self.candidate_idx)
        self.past1_idx: int = self.current_idx
        self.past2_idx: int = self.current_idx
        self.optimal_idx: int = optimal_idx
        self.action_space = ActionSpace(image_data_aug=image_data_aug, N=N)
        self.image_data_aug = image_data_aug
        # print(self.data)
        # exit()

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Reset the environment to the initial state
        # Example initial state

        self.current_idx = random.choice(self.candidate_idx)
        self.past1_idx: int = self.current_idx
        self.past2_idx: int = self.current_idx

        return self.get_state()

    def step(
        self, action_index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], float, bool]:
        # Apply the action and update the environment state

        self.action_space.generate_action_mapping(self.current_idx)
        action = self.action_space.get_action(self.current_idx, action_index)

        self.past2_idx = self.past1_idx
        self.past1_idx = self.current_idx
        self.current_idx = action

        # print([self.current_idx, self.past1_idx, self.past2_idx])

        # Calculate reward and check if the episode is done

        reward = self.calculate_reward()
        done = self.check_if_done()
        return self.get_state(), reward, done

    def get_image(self, idx: int) -> torch.Tensor:
        index_in_list = self.candidate_idx.index(idx)

        image = self.image_data_aug[index_in_list]["image"]
        if image is None:
            raise ValueError(f"No image data available for id={idx}")

        return image

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the current state of the environment

        image_current = self.get_image(self.current_idx)
        state_current, avg_luma_curr = feature_extractor(image_current)

        image_past1 = self.get_image(self.past1_idx)
        state_past1, avg_luma_past1 = feature_extractor(image_past1)

        image_past2 = self.get_image(self.past2_idx)
        state_past2, avg_luma_past2 = feature_extractor(image_past2)

        # state = torch.cat((state_current, state_past1, state_past2))

        # Just return the image features for now as state
        # state = torch.tensor(state_current)

        # Compute Flicker metric
        # if self.current_idx == self.past1_idx and self.past1_idx == self.past2_idx:
        #     flicker_metric = 1
        # else:
        #     flicker_metric = abs(self.current_idx - self.past2_idx) / (
        #         abs(self.current_idx - self.past1_idx)
        #         + abs(self.past1_idx - self.past2_idx)
        #     )

        # print(state.shape)

        # Combine state and flicker_metric into a single state vector
        # state = torch.cat((torch.tensor(state_current), torch.tensor([flicker_metric])))

        state_previous = torch.tensor([avg_luma_curr, avg_luma_past1, avg_luma_past2])

        return state_current, state_previous

    # def calculate_reward(self) -> float:
    #     # Define how the reward is calculated

    #     # image = self.get_image()

    #     optimal_idx = self.optimal_idx
    #     distance = abs(self.current_idx - optimal_idx)
    #     normalized_distance = distance / len(self.image_data["idx"])
    #     # Define a threshold for negative reward

    #     threshold = 0.2
    #     if self.current_idx == optimal_idx:
    #         reward = 1
    #     else:
    #         reward = 0
    #     # Modify reward based on distance

    #     if normalized_distance <= threshold:
    #         reward += 1 - normalized_distance  # Decrease reward based on distance
    #     else:
    #         if self.current_idx > optimal_idx:  # Over-exposure
    #             reward -= 2 * normalized_distance
    #         else:  # Under-exposure
    #             reward -= 2 * normalized_distance
    #     return reward  # Placeholder

    def calculate_reward(self) -> float:
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

        image = self.get_image(self.current_idx)
        image = image.float() / 255.0

        epsilon = 5 / 255  # Adjust epsilon as needed
        # Calculate stat rewards based on pre-defined conditions
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
        stat_reward = (
            (r1 * weight1)
            + (r2 * weight2)
            + (r3 * weight3)
            + (r4 * weight4)
            + (r5 * weight5)
        )
        # Compute Flicker reward
        if self.current_idx == self.past1_idx and self.past1_idx == self.past2_idx:
            flicker_metric = 1
        else:
            flicker_metric = abs(self.current_idx - self.past2_idx) / (
                abs(self.current_idx - self.past1_idx)
                + abs(self.past1_idx - self.past2_idx)
            )

        flicker_reward = flicker_metric
        reward = stat_reward + 5 * flicker_reward
        return reward

    def check_if_done(self) -> bool:
        # Define the termination condition

        return False  # Placeholder

    @property
    def number_state(self) -> int:
        # Assuming the state is [image_features, current_iso, current_ext]
        # and image_features is a list or array of features

        sample_state = self.get_state()
        num_features = len(sample_state)  # Number of features in the image
        return num_features
