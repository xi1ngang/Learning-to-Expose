from __future__ import absolute_import, division, print_function

import datetime

import io

import json
import os

import random
import time
from collections import deque
from itertools import count
from typing import Deque, NamedTuple

import numpy as np

import torch
import torch.optim as optim

from manifold.clients.python import ManifoldClient

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .DRL_dataloader import SceneDataset  # noqa

from .DRL_env import ActionSpace, CameraCaptureEnv
from .DRL_model import DQN
from .DRL_options import OptionsTrainer as Options

# from .DRL_replaybuffer import ReplayMemory, Transition
from .utils import (  # noqa
    dataset_aug,
    feature_extractor,
    find_optimal_idx,
    load_dataset,
    select_action,
)


class Transition(NamedTuple):
    state: list[float]
    action: int
    next_state: list[float]
    reward: float


class ReplayMemory(object):
    """
    A replay memory buffer for storing and sampling transitions.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the replay memory with a given capacity.

        Args:
            capacity (int): The maximum number of transitions to store in the buffer.
        """
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self, state: list[float], action: int, next_state: list[float], reward: float
    ) -> None:
        """
        Saves a transition into the replay memory.
        Args:
            state (list[float]): The current state.
            action (int): The action taken.
            next_state (list[float]): The next state.
            reward (float): The reward received.
        """
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            A list of sampled transitions.
        """

        return random.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        """
        Returns the current number of transitions stored in the replay memory.
        """
        return len(self.memory)


class Trainer:
    def __init__(self, options):
        self.epoch = 0

        self.opt = options

        self.episode_rewards = []  # List to store rewards from each episode

        self.episode_mean_q_values = []

        # setup device
        self._setup_devices()

        # setup model
        self._setup_model(
            self.opt.crop_size, 3, 2 * self.opt.slide_num + 1
        )  # 3 is the number of total frames considered

        # setup optimizer
        self._setup_optimizer()

        # setup criterion
        self._setup_criterion()

        # setup replay buffer
        self._setup_replay_buffer()

    def _save_opts(self):
        """Save all options to disk"""

        options_file = os.path.join(self.ckp_path, "commandline_args.json")
        with self.path_manager.open(options_file, mode="w") as f:
            json.dump(self.opt.__dict__.copy(), f, indent=2)

    def _setup_devices(self):
        """Setup training devices"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_cnt = 1 if self.opt.no_cuda else torch.cuda.device_count()

    def _setup_model(self, n_observations, n_previous_features, n_actions):
        """Setup model"""
        assert self.device

        self.model = None
        self.parameters_to_train = []

        # [Customization - Required]
        # Possible customizations:
        # * add model params in options.py and pass to model constructor
        self.policy_net = DQN(
            n_observations, n_previous_features, n_actions, self.opt
        ).to(self.device)
        self.target_net = DQN(
            n_observations, n_previous_features, n_actions, self.opt
        ).to(self.device)
        self.parameters_to_train += list(self.policy_net.parameters())

    def _setup_replay_buffer(self):
        """Setup replay buffer"""
        self.replay_buffer = None

        # [Customization - Required]
        # Possible customizations:
        # * add replay buffer params in options.py and pass to replay buffer constructor
        self.replay_buffer = ReplayMemory(self.opt.buffer_size)

    def _setup_optimizer(self):
        """Setup optimizer"""
        assert self.parameters_to_train

        self.model_optimizer = None
        self.model_lr_scheduler = None

        # [Customization - Option]
        # Possible customizations:
        # * change optimizer
        # * add optimizer/scheduler params in options.py and pass to optimizer and lr_scheduler
        self.model_optimizer = optim.AdamW(
            self.parameters_to_train, self.opt.learning_rate, amsgrad=True
        )
        # self.model_lr_scheduler = optim.lr_scheduler.StepLR(
        #     self.model_optimizer, self.opt.scheduler_step_size, 0.1
        # )

    def _setup_criterion(self):
        """Setup criterion"""
        assert self.device
        self.criterion = None

        # [Customization - Required]
        # Possible customizations:
        # * update criterion for dedicated applications
        # * add criterion params in options.py if need, and pass to criterion loss function
        self.criterion = torch.nn.SmoothL1Loss()
        self.criterion = self.criterion.to(self.device)

    def _set_train(self):
        """Convert all models to training mode"""
        self.model.train()

    def _set_eval(self):
        """Convert all models to testing/evaluation mode"""
        self.model.eval()

    def _save_model(self, episode_num=None):
        """Save the model or checkpoint."""
        path_suffix = f"trained_new_modelarti_higherflicker_wbfactor4/model_3500_nbd_{episode_num}_{self.opt.slide_num}_{self.opt.hidden_layer_size}.pt"
        manifold_path = f"pi_control/tree/{path_suffix}"
        bucket, path = manifold_path.split("/", 1)
        # Save the model state dict to a buffer
        buffer = io.BytesIO()
        torch.save(self.policy_net.state_dict(), buffer)
        buffer.seek(0)
        with ManifoldClient.get_client(bucket) as client:
            client.sync_put(path, buffer.getvalue())
        print(f"Episode {episode_num} model saved to {manifold_path}")

    def _save_rewards(self):
        """Save the episode rewards to a file."""
        manifold_path = "pi_control/tree"
        bucket, path = manifold_path.split("/", 1)
        # Save the episode rewards to a NumPy array file in Manifold
        rewards_path = f"{path}/trained_new_modelarti_higherflicker_wbfactor4/reward_3500_nbd_{self.opt.slide_num}_{self.opt.hidden_layer_size}.npy"
        with ManifoldClient.get_client(bucket) as client:
            client.sync_put(rewards_path, np.array(self.episode_rewards).tobytes())
        print(f"Rewards saved to {rewards_path}")

    def _run_epoch(self, steps_done, image_data):
        assert self.device

        # find the optimal index for each episode

        # optimal_idx = find_optimal_idx(image_data, self.opt.target_luma / 255)
        optimal_idx = 40

        # print(f"the optimal index is {optimal_idx}")

        total_reward = 0

        action_space = ActionSpace(image_data, N=self.opt.slide_num)
        env = CameraCaptureEnv(
            action_space, image_data, optimal_idx, N=self.opt.slide_num
        )
        state_curr, state_previous = env.reset()
        state = torch.cat((state_curr, state_previous), dim=0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

        current_indices = []
        current_reward = []

        for t in count():
            env.action_space.generate_action_mapping(env.current_idx)
            action, steps_done = select_action(
                state,
                self.policy_net,
                action_space,
                self.opt.EPS_END,
                self.opt.EPS_START,
                self.opt.EPS_DECAY,
                steps_done,
                self.device,
            )

            action_idx = action

            observation, reward, terminated = env.step(action_idx)
            total_reward += reward

            reward = torch.tensor([reward], device=self.device)
            done = terminated
            current_indices.append(env.current_idx)
            current_reward.append(reward)
            # Unpack the observation into current and previous states
            state_current, state_previous = observation

            if terminated:
                next_state = None
            else:
                next_state = torch.cat((state_current, state_previous), dim=0)
                next_state = torch.tensor(
                    next_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            # Store the transition in memory

            self.replay_buffer.push(state, action, next_state, reward)

            state = next_state

            # Perform one step of the optimization (on the policy network)

            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′

            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.opt.TAU + target_net_state_dict[key] * (1 - self.opt.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

            if t == self.opt.step_epoch:
                done = True

            if done:
                self.episode_rewards.append(total_reward)
                # self.episode_mean_q_values.append(mean_q_values / self.opt.step_epoch)
                break

        return steps_done, total_reward

    def _compute_loss(self, preds, targets):
        # [Customization - Required]
        # Possible customizations:
        # * update usage of criterion for dedicated applications
        # * add criterion params in options.py if need, and pass to criterion loss function
        losses = self.criterion(preds, targets)

        return losses

    def optimize_model(self):
        assert self.device

        if len(self.replay_buffer) < self.opt.batch_size:
            return
        transitions = self.replay_buffer.sample(self.opt.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # print(action_batch.shape)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        next_state_values = torch.zeros(self.opt.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values

        expected_state_action_values = (
            next_state_values * self.opt.gamma
        ) + reward_batch

        criterion = torch.nn.SmoothL1Loss()

        # criterion = nn.MSELoss()  # Replace with MSE loss function
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.model_optimizer.step()

    def train(self):
        """Train the model"""
        steps_done = 0
        num_moderate = 0
        num_bright = 0
        num_dim = 0
        scene_index = 0
        for num_epoch in range(self.opt.num_epochs + 1):
            if num_epoch % ((self.opt.num_epochs + 1) / 70) == 0:
                if num_moderate < 27:
                    current_scenario = "neutral"
                    num_moderate += 1
                    scene_index = num_moderate
                elif num_bright < 22:
                    current_scenario = "bright"
                    num_bright += 1
                    scene_index = num_bright
                elif num_dim < 21:
                    current_scenario = "dim"
                    num_dim += 1
                    scene_index = num_dim

                print(
                    f"Working on episode {num_epoch} and loading dataset for scenario {current_scenario} scene {scene_index}"
                )

                # Load the dataset for the current scenario
                manifold_path = f"pi_control/tree/ExpoSweep/training/{current_scenario}/{current_scenario}_scene{scene_index}"
                image_data = load_dataset(manifold_path)

            image_data_aug = dataset_aug(image_data, self.opt.crop_size)

            start = time.time()
            steps_done, total_reward = self._run_epoch(steps_done, image_data_aug)
            end = time.time()

            print(f"Episode {num_epoch} model saved")
            self._save_model(
                episode_num=num_epoch
            )  # Save checkpoint every 200 episodes

            print(
                f"Episode {num_epoch} done, time to run epoch: {end - start} seconds, total reward: {total_reward}"
            )

        self._save_rewards()  # Save the rewards after training


def main() -> None:
    options = Options()
    opts = options.parse()
    trainer = Trainer(opts)
    trainer.train()


if __name__ == "__main__":
    main()  # pragma: no cover
