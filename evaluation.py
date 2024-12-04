import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from plumEnv import PatrollingEnv

file_path = 'edge_matrix.csv'
plume_file_path = 'plume_matrix.csv'
def evaluate_model(env, model, num_episodes=100):
    """
    Evaluate the trained model by running it in the environment for a number of episodes.

    Args:
        env (gym.Env): The environment to run the simulation in.
        model (stable_baselines3.PPO): The trained PPO model.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        float: The mean number of timesteps before termination or truncation across all episodes.
    """
    total_timesteps = []
    for episode in range(num_episodes):
        obs = env.reset()  # Reset the environment at the start of each episode
        done = False
        timestep = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)  # Predict the action
            # print(action)
            obs, rewards, dones, info = env.step(action)  # Take a step in the environment
            env.render(mode = "custom")
            done = dones[0]  # Check if the episode has terminated
            #Get environment timesteps

            if env.get_attr('timestep')[0] >= env.get_attr('release_timestep')[0]:
                timestep+=1
            # env.render(mode = "custom" )  # Render the environment
        # print(f"Episode {episode + 1} completed in {timestep} timesteps")
        total_timesteps.append(timestep)

    mean_timesteps = np.mean(total_timesteps)
    return mean_timesteps


def run_evaluation(num_agents_range=range(1, 5), num_episodes=100):
    """
    Run the evaluation for multiple models with different numbers of agents.

    Args:
        num_agents_range (range): The range of agent counts for evaluation.
        num_episodes (int): The number of episodes to run for each model.
    """
    for num_agents in num_agents_range:
        print(f"************ Evaluating model with {num_agents} agents ************")
        # Load the trained model
        model = PPO.load(f"num_agents_{num_agents}_best_model/best_model.zip")

        # Create the environment
        env = PatrollingEnv(file_path, 'test', num_agents=num_agents, max_timesteps=50,
                            render_mode="custom")
        env = DummyVecEnv([lambda: env])


        # Evaluate the model
        mean_timesteps = evaluate_model(env, model, num_episodes)
        print(f"Mean timesteps for {num_agents} agents: {mean_timesteps:.2f}")


# Run the evaluation for different numbers of agents (1 to 4)
run_evaluation()
