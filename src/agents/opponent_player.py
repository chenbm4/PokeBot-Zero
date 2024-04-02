# Simple agent that chooses moves based on output of a trained model

import random
import numpy as np
from gymnasium import spaces

from poke_env.player.battle_order import BattleOrder
from poke_env.player import Player, ObsType
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData

from envs.rl_gym_env import RLGymEnv

class OpponentPlayer(Player):
    def __init__(self, *args, **kwargs):
        super(OpponentPlayer, self).__init__(*args, **kwargs)
        self.helper = RLGymEnv(opponent=None, battle_format="gen8randombattle")
        self.model = None

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.helper.calc_reward(last_battle, current_battle)
    
    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        return self.helper.embed_battle(battle)

    def describe_embedding(self) -> spaces.Space:
        return self.helper.describe_embedding()
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if self.model is not None:
            obs = self.embed_battle(battle)
            action, _ = self.model.predict(obs)
            return self.helper.action_to_move(action, battle)
        else:
            return random.choice(battle.available_moves)