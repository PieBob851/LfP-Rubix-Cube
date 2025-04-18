import torch
from torch import nn, zeros
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, enc_in_dim, layer_size=2048, latent_dim=256, epsilon=1e-4):
        super(Encoder, self).__init__()

        self.epsilon = epsilon

        self.lstm1 = nn.LSTM(enc_in_dim, layer_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(layer_size * 2, layer_size, batch_first=True, bidirectional=True)

        self.mu = nn.Linear(layer_size * 2, latent_dim)
        self.sigma = nn.Linear(layer_size * 2, latent_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        mu = self.mu(x[:, -1, :])
        sigma = F.softplus(self.sigma(x[:, -1, :])) + self.epsilon
        # mu = self.mu(x[-1, :])
        # sigma = F.softplus(self.sigma(x[-1, :])) + self.epsilon

        sample = torch.randn_like(sigma)
        z = mu + sigma * sample

        return z, mu, sigma