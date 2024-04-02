import asyncio
import time
import random
import tqdm
import os

from poke_env.player.battle_order import BattleOrder
import torch
import numpy as np
from torch import nn

import gymnasium as gym
from gymnasium import spaces

from poke_env.player import Player, RandomPlayer, Gen8EnvSinglePlayer, ObsType
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.data.gen_data import GenData

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from utils.trueskill_eval import RLSkillEvaluator


class TestPlayer(Gen8EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super(TestPlayer, self).__init__(*args, **kwargs)
        self.model = None

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

    def describe_embedding(self) -> spaces.Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return spaces.Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    

class OpponentPlayer(Player):
    def __init__(self, *args, **kwargs):
        super(OpponentPlayer, self).__init__(*args, **kwargs)
        self.helper = TestPlayer(opponent=None, battle_format="gen8randombattle")
        self.model = None

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

    def describe_embedding(self) -> spaces.Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return spaces.Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if self.model is not None:
            obs = self.embed_battle(battle)
            action, _ = self.model.predict(obs)
            return self.helper.action_to_move(action, battle)
        else:
            return random.choice(battle.available_moves)
    

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


def generate_random_account():
    return AccountConfiguration(
        username="user_" + str(random.randint(0, 1000000)),
        password=""
    )

async def main():
    monitor_dir = 'monitor_data'
    os.makedirs(monitor_dir, exist_ok=True)

    # Create a new opponent agent
    # opponent_account = generate_random_account()
    opponent_agent = OpponentPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    random_agent = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)

    train_agent = TestPlayer(
        battle_format="gen8randombattle", 
        # account_configuration=generate_random_account(),
        opponent=opponent_agent,
        start_challenging=True)
    
    eval_agent = TestPlayer(
        battle_format="gen8randombattle",
        opponent=random_agent,
        start_challenging=True)
    
    # Wrap the evaluation agent with the Monitor
    eval_env = Monitor(eval_agent, monitor_dir)

    # Initialize TrueSkill Evaluator
    trueskill_evaluator = RLSkillEvaluator()
    trueskill_evaluator.add_player('random')
    
    # Initialize models
    policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor)
    model = PPO("MlpPolicy", train_agent, verbose=1, policy_kwargs=policy_kwargs)
    opponent_model = PPO("MlpPolicy", train_agent, verbose=1, policy_kwargs=policy_kwargs)
    opponent_agent.model = opponent_model
    best_model_params = model.get_parameters()

    # best_performance = -float('inf')
    total_iterations = 5

    for iteration in range(total_iterations):
        # Set opponent parameters to the best model
        print("Setting opponent model parameters")
        opponent_model.set_parameters(best_model_params)
        opponent_agent.model = opponent_model
        # train_agent.start_challenging()

        # Train the model
        print("Training model")
        model.learn(total_timesteps=2048, progress_bar=True)
        print(f"Iteration {iteration + 1}/{total_iterations} complete")
        # Evaluate and update the best model every iteration
        print("Evaluating model")
        games_won = 0
        games_played = 10
        trueskill_evaluator.add_player(f'iteration_{iteration+1}')
        print("Results against random player:")
        for _ in tqdm.tqdm(range(games_played)):
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, render=False)
            # print(f"won {eval_env.n_won_battles} out of {eval_env.n_finished_battles} episodes")
            if eval_env.n_won_battles > 0:
                trueskill_evaluator.update_skills(f'iteration_{iteration+1}', 'random')
                games_won += 1
            else:
                trueskill_evaluator.update_skills('random', f'iteration_{iteration+1}')

            eval_env.reset_env(opponent=random_agent, restart=True)
        
        print(f"{games_won} victories out of {games_played} episodes")
        
        # games_won = 0
        # print(f"Results against {iteration} player:")
        # eval_env.reset_env(opponent=opponent_agent, restart=True)
        # for _ in tqdm.tqdm(range(games_played)):
        #     mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, render=False)
        #     if eval_env.n_won_battles > 0:
        #         trueskill_evaluator.update_skills(f'iteration_{iteration+1}', f'iteration_{iteration}')
        #         games_won += 1
        #     else:
        #         trueskill_evaluator.update_skills(f'iteration_{iteration}', f'iteration_{iteration+1}')

        #     eval_env.reset_env(opponent=opponent_agent, restart=True)
        
        # print(f"{games_won} victories out of {games_played} episodes")

        # Update the best model parameters
        best_model_params = model.get_parameters()
        
        # Reset the environment for future training
        print("Resetting environment")
        train_agent.reset_env(opponent=None, restart=True)

    # Save the final model
    model.save("final_model")

    # Save the final skill ratings
    trueskill_evaluator.save_leaderboard('leaderboards/ppo_leaderboard.csv')
    

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
        
