# Description: Feature extractor for the Pokemon environment.

import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Basic feature extractor for the Pokemon environment
# It is a simple neural network with two linear layers
# It processes the raw observations and returns a 10-dimensional feature vector
class BasicFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(BasicFeatureExtractor, self).__init__(observation_space, features_dim=10)
        self.net = nn.Sequential(
            nn.Linear(self._observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)