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
    def __init__(self, image_input_shape, joint_input_dim, num_actions, hidden_dim,
                 n_tasks, compression_dim=None, n_hidden_layers=1,
                 checkpoint_dir='checkpoints', name='bc_network'):
        super(Model, self).__init__()

        # Per-modality embedding width before fusion. Defaults to hidden_dim
        # (the historical behavior); pass a value to tune it independently.
        if compression_dim is None:
            compression_dim = hidden_dim

        self.conv1 = nn.Conv2d(image_input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *image_input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        self.joint_input = nn.Linear(joint_input_dim, compression_dim)

        self.image_input = nn.Linear(flat_size, compression_dim)

        # Which task to perform. In libero_spatial every task shares the scene
        # and differs only in the instruction, so without this the ten tasks are
        # the same observation with contradictory labels.
        # Indexed by raw task_id over the whole suite, so no id remapping exists
        # to get wrong. Equivalent to one-hot -> bias-free Linear; no relu,
        # because the embedding is already a free vector and clamping it to the
        # positive orthant would only cost capacity.
        self.task_input = nn.Embedding(n_tasks, compression_dim)

        self.compression_layer = nn.Linear(compression_dim * 3, hidden_dim)

        # n_hidden_layers hidden FC layers between fusion and output.
        # n_hidden_layers=1 reproduces the original single `linear1`.
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )

        self.output = nn.Linear(hidden_dim, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.apply(weights_init_)


    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def forward(self, obs, joint_state, task_id):
        x_image = self._conv_forward(obs)
        x_image = F.relu(self.image_input(x_image))

        x_joint = F.relu(self.joint_input(joint_state))

        x_task = self.task_input(task_id)

        x = torch.cat([x_image, x_joint, x_task], dim=1)

        x = F.relu(self.compression_layer(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.output(x))
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

