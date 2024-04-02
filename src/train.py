import os
import asyncio

from tensorboard import summary
from tqdm import tqdm
from tensorboardX import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from poke_env.player import RandomPlayer

from utils.config import load_config
from utils.trueskill_eval import RLSkillEvaluator
from envs.rl_gym_env import RLGymEnv
from agents.opponent_player import OpponentPlayer


async def main():
    config = load_config('src/utils/config.json')
    writer = SummaryWriter(log_dir=config['log_dir'])
    # test_env = RLGymEnv(opponent=None, battle_format="gen8randombattle")
    # check_env(test_env)

    # Initialize TrueSkill Evaluator
    trueskill_evaluator = RLSkillEvaluator()
    trueskill_evaluator.add_player('random')

    opponent_agent = OpponentPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    train_env = RLGymEnv(opponent=opponent_agent, 
                         battle_format="gen8randombattle", 
                         start_challenging=True)

    random_agent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = RLGymEnv(
        battle_format="gen8randombattle",
        opponent=random_agent,
        start_challenging=True)

    train_model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=config['log_dir'])
    opponent_model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=config['log_dir'])
    opponent_agent.model = opponent_model

    best_model_params = train_model.get_parameters()
    
    for iteration in range(config['n_iterations']):
        print(f'--- Iteration {iteration+1} ---')
        opponent_model.set_parameters(best_model_params)
        opponent_agent.model = opponent_model
        train_model.learn(total_timesteps=config['n_steps'])
        
        trueskill_evaluator.add_player(f'iteration_{iteration+1}')
        games_won = 0
        games_played = config['n_eval_games']
        for _ in tqdm(range(config['n_eval_games'])):
            mean_reward, _ = evaluate_policy(train_model, eval_env, n_eval_episodes=1, render=False)
            if eval_env.n_won_battles > 0:
                trueskill_evaluator.update_skills(f'iteration_{iteration+1}', 'random')
                games_won += 1
            else:
                trueskill_evaluator.update_skills('random', f'iteration_{iteration+1}')

            eval_env.reset_env(opponent=random_agent, restart=True)

        print(f"{games_won} victories out of {games_played} episodes. Mean reward: {mean_reward}")
        
        best_model_params = train_model.get_parameters()
        train_env.reset_env(opponent=opponent_agent, restart=True)
    
    trueskill_evaluator.save_leaderboard(os.path.join(config['log_dir'], 'leaderboard.json'))
    writer.close()
    train_model.save(os.path.join(config['model_dir'], 'final_model'))

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())