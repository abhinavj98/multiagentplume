
file_path = 'edge_matrix.csv'
# plume_file_path = 'plume_matrix.csv'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from plumEnv import PatrollingEnv
import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback
from policy import MultiInputActorCriticPolicy, MultiAgentPolicy
for num_agents in range(1, 5):
    print(f"************ Training with {num_agents} agents ************")
    env = PatrollingEnv(file_path, 'sims', num_agents=num_agents, max_timesteps=100, render_mode="rgb_array")
    check_env(env)
    env = DummyVecEnv([lambda: env])
    test_env = PatrollingEnv(file_path, 'sims', num_agents=num_agents, max_timesteps=100, render_mode="blah")
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecMonitor(test_env)
    eval_callback = EvalCallback(test_env, best_model_save_path=f"./num_agents_{num_agents}_best_model",
                                 log_path="./logs/", eval_freq=10000, n_eval_episodes=10,
                                 deterministic=False, render=False)
    # Create the PPO model with MultiInputPolicy
    model = PPO(MultiAgentPolicy, env, verbose=1, tensorboard_log=f"runs/{num_agents}_agents", policy_kwargs={"num_agents": num_agents}, device="auto")
    # Define the callback

    # Train the model, passing the callback to print rewards
    model.learn(total_timesteps=2000000, callback=eval_callback)
    # model.save(f"{num_agents}_agents")


