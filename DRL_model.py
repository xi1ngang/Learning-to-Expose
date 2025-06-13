import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_previous_features, n_actions, options):
        super(DQN, self).__init__()
        self.opt = options

        # Layers for the current image intensities
        self.layer1_current = nn.Linear(n_observations, self.opt.hidden_layer_size)
        self.layer2_current = nn.Linear(
            self.opt.hidden_layer_size, self.opt.hidden_layer_size
        )

        # Layers for the previous frames features
        self.layer1_previous = nn.Linear(n_previous_features, 16)
        self.layer2_previous = nn.Linear(16, 16)

        # Combined layer
        self.combined_layer = nn.Linear(
            self.opt.hidden_layer_size + 16, self.opt.hidden_layer_size + 16
        )

        # Output layer
        self.layer3 = nn.Linear(self.opt.hidden_layer_size + 16, n_actions)

    def forward(self, x):
        x_current = x[:, 0:128]
        x_previous = x[:, 128:]

        # print(x_current.shape)
        # print(x_previous.shape)

        # Process current frame features
        x_current = F.relu(self.layer1_current(x_current))
        x_current = F.relu(self.layer2_current(x_current))

        # Process previous frames features
        x_previous = F.relu(self.layer1_previous(x_previous))
        x_previous = F.relu(self.layer2_previous(x_previous))

        # Concatenate the outputs
        x_combined = torch.cat((x_current, x_previous), dim=1)

        # Process combined features
        x_combined = F.relu(self.combined_layer(x_combined))

        # Output layer
        x = self.layer3(x_combined)

        return x
