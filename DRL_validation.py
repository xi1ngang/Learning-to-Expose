import copy
import io
import json
import math
import random
from collections import deque, namedtuple
from itertools import count

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageOps

from .DRL_env import ActionSpace, CameraCaptureEnv
from .DRL_model import DQN
from .DRL_options import OptionsTrainer as Options

from .utils import (  # noqa
    calculate_reward,
    feature_extractor,
    load_dataset,
    resize_dataset,
)


# env = gym.make("CartPole-v1")

# set up matplotlib


is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display  # noqa
plt.ion()

# if GPU is to be used


class Validator:
    def __init__(self, options):
        self.opt = options

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def _reward_plot(self, reward_path):
        file_path = reward_path
        manifold_path = "pi_control/tree"
        bucket, path = manifold_path.split("/", 1)
        # Load the episode rewards from the NumPy array file in Manifold

        with ManifoldClient.get_client(bucket) as client:
            buffer = io.BytesIO()
            client.sync_get(file_path, buffer)
            buffer.seek(0)
            reward_list = np.frombuffer(buffer.read(), dtype=np.float64)
        # Plotting
        reward_list = pd.Series(reward_list)
        moving_avg = reward_list.rolling(1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg, color="blue", linewidth=2.0)
        # plt.fill_between(range(len(reward)), mean_rewards - 2 * std_rewards, mean_rewards + 2 * std_rewards, color = "blue", alpha=0.2)

        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        # plt.title('Average Reward Trajectory with Variability Shading')

        plt.legend()
        print("I am plotting the reward.")
        plt.savefig(
            "/data/sandcastle/boxes/fbsource/fbcode/pi_algo/projects/DRL_AEC_w_newfeature/outputs/reward_flicker_1400_nbd_plot2.png"
        )

    def _model_eval_inference(
        self,
        model,
        val_image_data_aug,
        start_index,
        device,
        action_space,
        max_steps=100,
    ):
        """
        Validate a DQN model by iterating through the environment until convergence or reaching the maximum number of steps.
        Args:
            model (nn.Module): The DQN model to be validated.
            val_image_data_aug (list): A list of augmented validation images.
            start_index (int): The starting index for the validation process.
            device (torch.device): The device to run the model on.
            max_steps (int): Maximum number of steps to validate before stopping.
        Returns:
            is_converge (bool): Whether the model has converged.
            steps2converge (int): The number of steps taken to converge.
            moving_index_list (list): A list of indices visited during the validation process.
            images (list): Collected images during validation.
            features (list): Collected features during validation.
            Reward (list): Collected rewards during validation.
        """

        # Initialize variables to track the validation process

        steps2converge = 0
        moving_index_list = [start_index]
        curr_index = start_index
        past1_index = start_index
        past2_index = start_index
        is_converge = False
        # Prepare to collect images and features for plotting

        Reward = []

        while steps2converge < max_steps:
            # Extract the current image and its features
            image = val_image_data_aug[curr_index]["image"]
            image_tensor = torch.tensor(image)
            stat_state, feature = feature_extractor(image_tensor)

            # if curr_index == past1_index and past1_index == past2_index:
            #     flicker_state = 1
            # else:
            #     flicker_state = abs(curr_index - past2_index) / (
            #         abs(curr_index - past1_index) + abs(past1_index - past2_index)
            #     )

            # Combine state and flicker_metric into a single state vector
            # state = torch.cat((torch.tensor(stat_state), torch.tensor([flicker_state])))
            state = torch.tensor(stat_state)

            # Prepare state tensor based on the number of frames
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            reward = calculate_reward(image)
            # Collect image and feature for later plotting
            Reward.append(reward)
            # Pass the tensor to the model and get the best action
            state_action_value = model(state_tensor)
            best_action_index = torch.argmax(state_action_value)
            # Get the next index based on the best action
            action_space.generate_action_mapping(curr_index)
            next_index = action_space.get_action(curr_index, best_action_index)
            steps2converge += 1
            moving_index_list.append(next_index)
            # Check for convergence
            if abs(next_index - curr_index) <= 1:
                is_converge = True
                break
            curr_index = next_index
            past2_index = past1_index
            past1_index = curr_index
        return is_converge, steps2converge, moving_index_list, Reward

    def _model_eval_wflicker_inference(
        self,
        model,
        val_image_data_aug,
        start_index,
        device,
        action_space,
        max_steps=100,
    ):
        """
        Validate a DQN model by iterating through the environment until convergence or reaching the maximum number of steps.
        Args:
            model (nn.Module): The DQN model to be validated.
            val_image_data_aug (list): A list of augmented validation images.
            start_index (int): The starting index for the validation process.
            device (torch.device): The device to run the model on.
            max_steps (int): Maximum number of steps to validate before stopping.
        Returns:
            is_converge (bool): Whether the model has converged.
            steps2converge (int): The number of steps taken to converge.
            moving_index_list (list): A list of indices visited during the validation process.
            images (list): Collected images during validation.
            features (list): Collected features during validation.
            Reward (list): Collected rewards during validation.
        """

        # Initialize variables to track the validation process

        steps2converge = 0
        moving_index_list = [start_index]
        curr_index = start_index
        past1_index = start_index
        past2_index = start_index
        is_converge = False
        # Prepare to collect images and features for plotting

        Reward = []

        while steps2converge < max_steps:
            # Extract the current image and its features
            image = val_image_data_aug[curr_index]["image"]
            image_past1 = val_image_data_aug[past1_index]["image"]
            image_past2 = val_image_data_aug[past2_index]["image"]
            image_tensor = torch.tensor(image)
            state_curr, feature1 = feature_extractor(image_tensor)
            _, feature2 = feature_extractor(torch.tensor(image_past1))
            _, feature3 = feature_extractor(torch.tensor(image_past2))
            # Prepare state tensor based on the number of frames
            state_previous = torch.tensor([feature1, feature2, feature3])
            state = torch.cat((state_curr, state_previous), dim=0)
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            reward = calculate_reward(image)
            # Collect image and feature for later plotting
            Reward.append(reward)
            # Pass the tensor to the model and get the best action
            state_action_value = model(state_tensor)
            best_action_index = torch.argmax(state_action_value)
            # Get the next index based on the best action
            action_space.generate_action_mapping(curr_index)
            next_index = action_space.get_action(curr_index, best_action_index)
            steps2converge += 1
            moving_index_list.append(next_index)
            # Check for convergence
            if abs(next_index - curr_index) <= 1:
                is_converge = True
                break
            curr_index = next_index
            past2_index = past1_index
            past1_index = curr_index
        return is_converge, steps2converge, moving_index_list, Reward

    def evaluation_outdoor(self):
        # Load validation dataset
        scenario = "outdoor"
        scene_index = 1

        for scene_index in range(1, 16):
            print(f"Evaluating scenario {scenario} and scene {scene_index}")
            manifold_path = f"pi_control/tree/ExpoSweep/evaluation/outdoor/{scenario}_scene{scene_index}"
            val_image_data = load_dataset(manifold_path)
            dataset_resize = resize_dataset(val_image_data, re_size=self.opt.crop_size)

            # episode_num = range(0, 2100)
            # episode_num = [1140]
            episode_num = range(1000, 1400)
            slide_num = self.opt.slide_num

            average_rewards = []
            average_steps = []
            all_results = []

            for episode in episode_num:
                action_space = ActionSpace(dataset_resize, N=slide_num)
                n_actions = 2 * slide_num + 1
                n_observations = self.opt.crop_size  # plus flicker metric
                manifold_path = "pi_control/tree"
                bucket, path = manifold_path.split("/", 1)
                with ManifoldClient.get_client(bucket) as client:
                    buffer = io.BytesIO()
                    # client.sync_get(
                    #     f"{path}/dim_128/model_1400_nbd_{episode}_{self.opt.slide_num}_{self.opt.hidden_layer_size}.pt",
                    #     buffer,
                    # )
                    client.sync_get(
                        f"{path}/trained_new_modelarti_higherflicker_wbfactor2/model_1400_nbd_{episode}_{self.opt.slide_num}_{self.opt.hidden_layer_size}.pt",
                        buffer,
                    )

                    buffer.seek(0)
                    loaded_state_dict = torch.load(
                        buffer, map_location=torch.device("cpu")
                    )
                # Assuming you have your model architecture defined elsewhere

                model = DQN(n_observations, 3, n_actions, self.opt)
                model.load_state_dict(loaded_state_dict)
                model.to(self.device)

                # Run validation
                # num_trials = 50
                is_convergence = []
                steps = []
                reward = []
                moving_index = []
                start_indexes = [4, 51, 87]

                for start_index in start_indexes:
                    # Randomly select a start index
                    # start_index = random.randint(0, 110)
                    # Validate the model
                    (
                        is_converge,
                        steps2converge,
                        moving_index_list,
                        rewards,
                    ) = self._model_eval_wflicker_inference(
                        model,
                        dataset_resize,
                        start_index,
                        self.device,
                        action_space,
                        max_steps=100,
                    )
                    is_convergence.append(is_converge)
                    steps.append(steps2converge)
                    reward.append(rewards[-1])
                    moving_index.append(moving_index_list)

                if np.mean(is_convergence) < 1:
                    steps = 0
                avg_reward = np.mean(reward)
                avg_steps = np.mean(steps)
                average_rewards.append(avg_reward)
                average_steps.append(avg_steps)

                results = {
                    "episode_number": episode,
                    "moving_index_list": moving_index,
                    "reward_sequences": reward,
                }

                if np.mean(reward) > 0:
                    all_results.append(results)
                # all_results.append(results)

                # if np.mean(is_convergence) == 1 and np.mean(reward) >= 0.5:
                #     all_results.append(results)

                print(
                    f"Episode {episode}, is_convergence {np.mean(is_convergence)}, moving index is {moving_index}, average convergence steps {np.mean(steps)}, average reward {np.mean(reward)}"
                )

            # with open(
            #     f"/data/sandcastle/boxes/fbsource/fbcode/pi_algo/projects/DRL_AEC_w_newfeature/outputs/higher_flicker_wfactor_1140/results_flicker_{episode}_{scenario}_scene{scene_index}.json",
            #     "w",
            # ) as f:
            #     json.dump(all_results, f, indent=4)

    def evaluation(self):
        # Load validation dataset
        scenarios = ["dim", "neutral", "bright"]
        scene_index = 1

        for scenario in scenarios:
            for scene_index in range(1, 6):
                print(f"Evaluating scenario {scenario} and scene {scene_index}")
                manifold_path = f"pi_control/tree/ExpoSweep/evaluation/indoor/{scenario}_scene{scene_index}"
                val_image_data = load_dataset(manifold_path)
                dataset_resize = resize_dataset(
                    val_image_data, re_size=self.opt.crop_size
                )

                # episode_num = range(1000, 1400)
                # episode_num = range(1000, 1800)
                episode_num = [1140]
                slide_num = self.opt.slide_num

                average_rewards = []
                average_steps = []
                all_results = []

                for episode in episode_num:
                    action_space = ActionSpace(dataset_resize, N=slide_num)
                    n_actions = 2 * slide_num + 1
                    n_observations = self.opt.crop_size
                    manifold_path = "pi_control/tree"
                    bucket, path = manifold_path.split("/", 1)
                    with ManifoldClient.get_client(bucket) as client:
                        buffer = io.BytesIO()
                        # client.sync_get(
                        #     f"{path}/trained_2100_model/model_2100_nbd_{episode}_{self.opt.slide_num}_{self.opt.hidden_layer_size}.pt",
                        #     buffer,
                        # )

                        client.sync_get(
                            f"{path}/trained_new_modelarti_higherflicker/model_1400_nbd_{episode}_{self.opt.slide_num}_{self.opt.hidden_layer_size}.pt",
                            buffer,
                        )

                        buffer.seek(0)
                        loaded_state_dict = torch.load(
                            buffer, map_location=torch.device("cpu")
                        )
                    # Assuming you have your model architecture defined elsewhere

                    model = DQN(n_observations, 3, n_actions, self.opt)
                    model.load_state_dict(loaded_state_dict)
                    model.to(self.device)

                    # Run validation
                    # num_trials = 50
                    is_convergence = []
                    steps = []
                    reward = []
                    moving_index = []
                    start_indexes = [4, 51, 87]

                    for start_index in start_indexes:
                        # Randomly select a start index
                        # start_index = random.randint(0, 110)
                        # Validate the model
                        (
                            is_converge,
                            steps2converge,
                            moving_index_list,
                            rewards,
                        ) = self._model_eval_wflicker_inference(
                            model,
                            dataset_resize,
                            start_index,
                            self.device,
                            action_space,
                            max_steps=50,
                        )
                        is_convergence.append(is_converge)
                        steps.append(steps2converge)
                        reward.append(rewards[-1])
                        moving_index.append(moving_index_list)

                    if np.mean(is_convergence) < 1:
                        steps = 0
                    avg_reward = np.mean(reward)
                    avg_steps = np.mean(steps)
                    average_rewards.append(avg_reward)
                    average_steps.append(avg_steps)

                    results = {
                        "episode_number": episode,
                        "moving_index_list": moving_index,
                        "reward_sequences": reward,
                    }
                    all_results.append(results)

                    if np.mean(is_convergence) == 1 and np.mean(reward) >= 0.8:
                        all_results.append(results)

                    print(
                        f"Episode {episode}, is_convergence {np.mean(is_convergence)}, moving index is {moving_index}, average convergence steps {np.mean(steps)}, average reward {np.mean(reward)}"
                    )

                with open(
                    f"/data/sandcastle/boxes/fbsource/fbcode/pi_algo/projects/DRL_AEC_w_newfeature/outputs/higher_flicker/results_higherflicker_{scenario}_scene{scene_index}.json",
                    "w",
                ) as f:
                    json.dump(all_results, f, indent=4)


def main() -> None:
    options = Options()
    opts = options.parse()
    validator = Validator(opts)
    # validator._reward_plot(
    #     "tree/trained_new_modelarti_higherflicker_wbfactor2/reward_1400_nbd_40_128.npy"
    # )
    validator.evaluation_outdoor()
    # validator.evaluation()


if __name__ == "__main__":
    main()  # pragma: no cover
