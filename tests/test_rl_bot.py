import unittest
from unittest.mock import MagicMock

from src.rl_bot import CustomFeatureExtractor, SimpleRLPlayer


class TestSimpleRLPlayer(unittest.TestCase):
    def test_calc_reward(self):
        # Create a SimpleRLPlayer instance
        player = SimpleRLPlayer()

        # Create mock battles for testing
        last_battle = MagicMock()
        current_battle = MagicMock()

        # Test the calc_reward method
        reward = player.calc_reward(last_battle, current_battle)

        # Assert that the reward is a float
        self.assertIsInstance(reward, float)

        # Add more test cases for different scenarios

    def test_embed_battle(self):
        # Create a SimpleRLPlayer instance
        player = SimpleRLPlayer()

        # Create a mock battle for testing
        battle = MagicMock()

        # Test the embed_battle method
        embedding = player.embed_battle(battle)

        # Assert that the embedding is a numpy array
        self.assertIsInstance(embedding, np.ndarray)

        # Add more test cases for different scenarios

class TestCustomFeatureExtractor(unittest.TestCase):
    def test_forward(self):
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
