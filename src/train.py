import os
import asyncio

from tensorboard import summary
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
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
    log_dir = os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize TrueSkill Evaluator
    trueskill_evaluator = RLSkillEvaluator()
    trueskill_evaluator.add_player('random')

    opponent_agent = OpponentPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    train_env = RLGymEnv(opponent=opponent_agent,
                         trueskill_evaluator=trueskill_evaluator,
                         player_name='ai_initial',
                         opponent_name='opponent_initial',
                         battle_format="gen8randombattle",
                         start_challenging=True)

    random_agent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = RLGymEnv(
        battle_format="gen8randombattle",
        opponent=random_agent,
        start_challenging=True)

    train_model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir)
    opponent_model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir)
    opponent_agent.model = opponent_model

    best_model_params = train_model.get_parameters()
    
    for iteration in range(config['n_iterations']):
        print(f'--- Iteration {iteration+1} ---')
        training_player_name = f'iteration_{iteration+1}'
        opponent_player_name = f'iteration_{iteration}'
        eval_player_name = 'random'

        opponent_model.set_parameters(best_model_params)
        train_env.set_trueskill_players(
            player_name=training_player_name, 
            opponent_name=opponent_player_name)
        train_model.learn(total_timesteps=config['n_steps'], tb_log_name='model', reset_num_timesteps=False)
        writer.add_scalar('Win Rate vs Previous Opponent', train_env.win_rate, iteration+1)

        games_won = 0
        games_played = config['n_eval_games']
        total_reward = 0
        for _ in tqdm(range(config['n_eval_games'])):
            mean_reward, _ = evaluate_policy(train_model, eval_env, n_eval_episodes=1, render=False)
            total_reward += mean_reward
            if eval_env.n_won_battles > 0:
                trueskill_evaluator.update_skills(training_player_name, eval_player_name)
                games_won += 1
            else:
                trueskill_evaluator.update_skills(eval_player_name, training_player_name)

            
            eval_env.reset_env(opponent=random_agent, restart=True)

        ai_skill = trueskill_evaluator.get_player_skill(opponent_player_name)
        writer.add_scalar('TrueSkill/AI Rating', ai_skill.mu, iteration)
        # Logging metrics
        writer.add_scalar('Win Rate vs Random', games_won / games_played, iteration+1)

        print(f"{games_won} victories out of {games_played} episodes. Mean reward: {total_reward/games_played}")

        best_model_params = train_model.get_parameters()

    trueskill_evaluator.save_leaderboard(os.path.join(log_dir, 'leaderboard.json'))
    writer.close()
    train_model.save(os.path.join(config['model_dir'], 'final_model'))

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
