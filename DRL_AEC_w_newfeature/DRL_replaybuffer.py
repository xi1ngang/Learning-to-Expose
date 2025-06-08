# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import random
from collections import deque
from typing import Deque, NamedTuple


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
