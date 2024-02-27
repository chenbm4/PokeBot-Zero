import unittest
import numpy as np
import torch
import gym
from stable_baselines3.common.evaluation import evaluate_policy
import tabulate
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    cross_evaluate,
    evaluate_player
)
from unittest.mock import MagicMock
import numpy as np
import torch
import gym
from stable_baselines3.common.evaluation import evaluate_policy

from src.rl_bot import CustomFeatureExtractor, SimpleRLPlayer
from unittest.mock import MagicMock

import tabulate

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    cross_evaluate,
    evaluate_player,
)


class TestSimpleRLPlayer(unittest.TestCase):
    def test_calc_reward_edge_cases(self):
        # Create a SimpleRLPlayer instance
        player = SimpleRLPlayer()

        # Create mock battles for testing
        last_battle = MagicMock()
        current_battle = MagicMock()

        # Test the calc_reward method
        reward = player.calc_reward(last_battle, current_battle)

        # Assert that the reward is a float
        self.assertIsInstance(reward, float)

        # Test the calc_reward method for different scenarios and edge cases
        # Reward for fainted_value = 0, hp_value = 0, victory_value = 0
        reward_1 = player.calc_reward(last_battle, current_battle)
        self.assertEqual(reward_1, 0.0)

        # Reward for fainted_value = 2.0, hp_value = 1.0, victory_value = 30.0
        reward_2 = player.calc_reward(last_battle, current_battle)
        self.assertEqual(reward_2, 2.0)

    def test_embed_battle(self):
        # Create a SimpleRLPlayer instance
        player = SimpleRLPlayer()

        # Create a mock battle for testing
        battle = MagicMock()

        # Test the embed_battle method
        embedding = player.embed_battle(battle)

        # Assert that the embedding is a numpy array
        self.assertIsInstance(embedding, np.ndarray)

        # Test the forward method for different scenarios and edge cases

class TestCustomFeatureExtractor(unittest.TestCase):
    def test_forward_edge_cases(self):
        # Create a CustomFeatureExtractor instance
        extractor = CustomFeatureExtractor()

        # Create a mock observation for testing
        observation = MagicMock()

        # Test the forward method
        output = extractor.forward(observation)

        # Assert that the output is a torch tensor
        self.assertIsInstance(output, torch.Tensor)

        # Add more test cases for different scenarios

if __name__ == "__main__":
    unittest.main()
