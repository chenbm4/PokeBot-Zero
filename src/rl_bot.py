import gym, torch
import asyncio
import numpy as np
from gymnasium.spaces import Box, Space
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

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


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=10)
        self.net = nn.Sequential(
            nn.Linear(self._observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


async def test_environment():
    # Test the environment to ensure the class is consistent with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    test_env = SimpleRLPlayer(battle_format="gen8randombattle", start_challenging=True, opponent=opponent)
    check_env(test_env)
    test_env.close()


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    create_environments()

def create_environments():
    # Function to create training and evaluation environments
    """
    Create training and evaluation environments for the RL model.
    """
    opponent = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    # Move the existing code here
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    """
Evaluate the RL model against a random player.

:param model: The RL model to be evaluated.
:param eval_env: The evaluation environment.
:param n_eval_episodes: The number of evaluation episodes.
:param render: Whether to render the evaluation.
"""
    opponent = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Creating the model
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
    )

    model = DQN(
        "MlpPolicy",
        train_env, 
        policy_kwargs=policy_kwargs, 
        learning_rate=0.00025, 
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.5,
        train_freq=4,
        gradient_steps=1,
        optimize_memory_usage=False,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
    return model

def train_model\(\):
    )

    # Training the model
    model.learn(total_timesteps=10000, )
    train_env.close()

    # Evaluating the model against random player
    print("Results against random player:")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=100, render=False)
evaluate_model_random

    # Evaluating against max base power player
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=100, render=False)
    print(f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes")
    eval_env.reset_env(restart=False)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
