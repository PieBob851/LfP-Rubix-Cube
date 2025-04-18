import torch
from torch import nn, zeros

class Planner(nn.Module):
    def __init__(self, obs_dim, goal_dim, layer_size=2048, latent_dim=256, epsilon=1e-4):
        super(Planner, self).__init__()

        self.epsilon = epsilon

        input_dim = obs_dim + goal_dim
        self.fc1 = nn.Linear(input_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size)

        self.mu = nn.Linear(layer_size, latent_dim)
        self.sigma = nn.Linear(layer_size, latent_dim)

    def forward(self, obs_init, obs_goal):
        x = torch.cat([obs_init, obs_goal], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + self.epsilon

        return mu, sigma