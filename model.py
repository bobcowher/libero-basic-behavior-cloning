import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='bc_network'):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self._conv_forward(dummy).shape[1]
        
        self.linear1 = nn.Linear(flat_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, num_actions)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.apply(weights_init_)


    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def forward(self, obs, joint_state):
        x = self._conv_forward(obs)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.output(x))
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

