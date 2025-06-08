import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, options):
        super(DQN, self).__init__()
        self.opt = options
        self.layer1 = nn.Linear(n_observations, self.opt.hidden_layer_size)
        self.layer2 = nn.Linear(self.opt.hidden_layer_size, self.opt.hidden_layer_size)
        self.layer3 = nn.Linear(self.opt.hidden_layer_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
