import torch
from torch import nn, zeros

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, goal_dim, layer_size=1024, latent_dim=256):
        super(Actor, self).__init__()

        input_dim = obs_dim + latent_dim + goal_dim
        self.lstm1 = nn.LSTM(input_dim, layer_size, batch_first=True)
        self.lstm2 = nn.LSTM(layer_size, layer_size, batch_first=True)

        self.actions = nn.Linear(layer_size, act_dim)
    # def forward(self, obs, latent_plan, goal, hidden_state=None):
    #     x = torch.cat([obs, latent_plan, goal], dim=-1)
    def forward(self, obs, z, goal, hidden_state=None):
        x = torch.cat([obs, z, goal], dim=-1)

        x, hidden_state = self.lstm1(x, hidden_state)
        x, hidden_state = self.lstm2(x, hidden_state)

        return self.actions(x), hidden_state