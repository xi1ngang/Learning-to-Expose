from __future__ import absolute_import, division, print_function

import argparse

from pi_algo.infra.fblearner.options_base import OptionsTrainerBase


class OptionsTrainer(OptionsTrainerBase):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DRL_AEC options")

        # TRAINING options
        # self.parser.add_argument("--resume", help="if resume", action="store_true")
        # ... more argument about
        # model params, such as arch, version

        # OPTIMIZATION options
        self.parser.add_argument(
            "--learning_rate", type=float, help="learning rate", default=1e-6
        )
        self.parser.add_argument(
            "--num_epochs", type=int, help="number of epochs", default=1400 - 1
        )
        self.parser.add_argument(
            "--hidden_layer_size", type=int, help="hidden layer size", default=64
        )
        self.parser.add_argument(
            "--train_update", type=int, help="number of epochs", default=250
        )
        self.parser.add_argument(
            "--step_epoch", type=int, help="training steps per epoch", default=2000
        )
        self.parser.add_argument(
            "--batch_size", type=int, help="batch size", default=512
        )
        self.parser.add_argument(
            "--gamma", type=float, help="discount factor", default=0.9
        )
        self.parser.add_argument(
            "--EPS_START", type=float, help="starting value of epsilon", default=0.9
        )
        self.parser.add_argument(
            "--EPS_END", type=float, help="final value of epsilon", default=0.05
        )
        self.parser.add_argument(
            "--EPS_DECAY",
            type=int,
            help="rate of exponential decay of epsilon, higher means a slower decay",
            default=10000,
        )
        self.parser.add_argument(
            "--TAU", type=float, help="update rate of the target network", default=0.005
        )
        self.parser.add_argument(
            "--buffer_size", type=int, help="replay memory buffer size", default=100000
        )

        # # dataset options
        self.parser.add_argument(
            "--crop_size",
            type=int,
            help="crop size for image augumentation",
            default=128,
        )

        self.parser.add_argument(
            "--target_luma",
            type=int,
            help="target luma for RL agent",
            default=80,
        )

        # # action space options
        self.parser.add_argument(
            "--slide_num",
            type=int,
            help="side window size for action space",
            default=40,
        )

        # SYSTEM options
        self.parser.add_argument(
            "--no_cuda", help="if set disables CUDA", action="store_true"
        )

        # LOGGING options
        self.parser.add_argument(
            "--log_frequency",
            type=int,
            help="number of batches between each tensorboard log",
            default=5,
        )
        self.parser.add_argument(
            "--save_frequency",
            type=int,
            help="number of epochs between each save",
            default=5,
        )
        self.parser.add_argument(
            "--val_frequency",
            type=int,
            help="number of epochs between each val",
            default=10,
        )
        self.parser.add_argument(
            "--export_frequency",
            type=int,
            help="number of epochs between each model export",
            default=10,
        )

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
