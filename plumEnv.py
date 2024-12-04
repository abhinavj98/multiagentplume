


import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import os
# Helper Functions
def process_grid(grid_file):
    grid_df = pd.read_csv(grid_file)
    edge_matrix = grid_df.iloc[:, 1:].values  # Exclude the first column for the edge matrix
    state_labels = grid_df.iloc[:, 0].apply(lambda s: tuple(map(int, map(float, s.strip('()').split(','))))).tolist()
    return edge_matrix, state_labels


def process_plume_matrix(plume_file):
    plume_df = pd.read_csv(plume_file)
    plume_df['parsed_index'] = plume_df['index'].apply(
        lambda s: tuple(map(int, s.strip('()').split(',')))
    )
    plume_df['parsed_index'] = plume_df['parsed_index'].apply(
        lambda x: ((x[0] - 1) * 10, (x[1] - 1) * 10, x[3] - 1)
    )
    plume_df['parsed_index'] = plume_df['parsed_index'].apply(
        lambda x: (round(x[0] / 35) * 35, round(x[1] / 35) * 35, x[2])
    )
    plume_dict = dict(zip(plume_df['parsed_index'], plume_df['val']))
    return plume_dict


class PatrollingEnv(gymnasium.Env):
    def __init__(self, grid_file, plume_folder, num_agents, max_timesteps, render_mode="rgb_array"):
        super(PatrollingEnv, self).__init__()

        # Initialize grid and plume data
        self.grid, self.state_labels = process_grid(grid_file)
        self.grid_size = len(self.grid)
        # self.plume_data = process_plume_matrix(plume_file)
        self.plume_files = glob(os.path.join(plume_folder, '*'))
        assert len(self.plume_files) > 0, "No plume files found in the specified folder."

        # Environment properties
        self.num_agents = num_agents
        self.max_timesteps = max_timesteps
        self.timestep = 0
        self.release_timestep = 0
        self.global_timestep = 0
        self.plume_sim = self.plume_files[0]
        self.wind_direction = 0.0
        self.wind_speed = 0.0
        self.render_mode = render_mode

        # Tensorboard
        self.writer = SummaryWriter(f"runs/{num_agents}_agents")
        self.ep_rew = 0
        self.episode = 0

        # Initialize agent positions
        self.agent_positions = {
             f'agent_{agent_id}': self.state_labels[np.random.randint(0, len(self.state_labels))]
            for agent_id in range(self.num_agents)
        }

        # Define action space for each agent
        self.action_space = spaces.MultiDiscrete([4]*self.num_agents)


        #     spaces.Dict({
        #     f'agent_{agent_id}': spaces.Discrete(5)  # 5 actions: up, down, left, right, stay
        #     for agent_id in range(self.num_agents)
        # }))

        # Define observation space dynamically for each agent
        agent_obs_space = {}
        for agent_id in range(self.num_agents):
            agent_obs_space[f'agent_{agent_id}_position'] = spaces.Discrete(len(self.state_labels))
            agent_obs_space[f'agent_{agent_id}_wind_direction'] = spaces.MultiBinary(8)  # 8 possible directions
            agent_obs_space[f'agent_{agent_id}_wind_speed'] = spaces.MultiBinary(2)  # 3 possible wind speeds
            agent_obs_space[f'agent_{agent_id}_plume_concentration'] = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            agent_obs_space[f'agent_{agent_id}_edge_info'] = spaces.MultiBinary(4)  # 4 possible directions

        self.observation_space = spaces.Dict(agent_obs_space)

    def encode_wind_direction(self, direction, num_bins=8):
        """Encodes wind direction into one-hot vector."""
        bin_index = int(direction // (360 / num_bins)) % num_bins
        one_hot = np.zeros(num_bins, dtype=np.float32)
        one_hot[bin_index] = 1.0
        return one_hot

    def encode_wind_speed(self, speed, bins=[0, 3, np.inf]):
        """Encodes wind speed into one-hot vector."""
        bin_index = np.digitize(speed, bins) - 1
        one_hot = np.zeros(len(bins) - 1, dtype=np.float32)
        one_hot[bin_index] = 1.0
        return one_hot

    def reset(self, seed=None):
        # Log the episode reward
        self.writer.add_scalar('Rewards', self.ep_rew, self.global_timestep)
        self.ep_rew = 0
        self.episode += 1
        self.release_timestep = np.random.randint(0, self.max_timesteps/2)
        self.plume_sim = np.random.choice(self.plume_files)
        #plume sim is of format sims/seattle_WD90_WS2_spaceneedle
        #WD is wind direction, WS is wind speed
        self.wind_direction = self.encode_wind_direction(float(self.plume_sim.split('_')[1][2:]))
        self.wind_speed = self.encode_wind_speed(float(self.plume_sim.split('_')[2][2:]))

        plume_sim_file = os.path.join(self.plume_sim, 'inner', 'QP_confield_sparse.csv')
        # print(f"Resetting environment for episode {self.episode} with plume simulation: {plume_sim_file}")
        self.plume_data = process_plume_matrix(plume_sim_file)

        # Seed the environment
        if seed is not None:
            np.random.seed(seed)

        self.timestep = 0
        self.agent_positions = {
            f'agent_{agent_id}': self.state_labels[np.random.randint(0, len(self.state_labels))]
            for agent_id in range(self.num_agents)
        }

        observations = self._get_obs()
        return observations, self._get_info()

    def _get_info(self):
        return {}

    def _get_obs(self):
        observations = {}
        for agent_id, position in self.agent_positions.items():
            current_index = self.state_labels.index(position)
            #Observations must be a dictionary with value as numpy array
            # Position, wind, plume concentration
            # observations[f'{agent_id}_position'] = current_index
            # observations[f'{agent_id}_wind_direction'] = self.wind_direction
            # observations[f'{agent_id}_wind_speed'] = self.wind_speed
            # observations[f'{agent_id}_plume_concentration'] = self.get_plume_value(
            #     position, time=max(0, self.timestep - self.release_timestep)
            # )
            observations[f'{agent_id}_position'] = current_index
            observations[f'{agent_id}_wind_direction'] = np.array(self.wind_direction).astype(np.int8)
            observations[f'{agent_id}_wind_speed'] = np.array(self.wind_speed).astype(np.int8)
            observations[f'{agent_id}_plume_concentration'] = np.array([self.get_plume_value(
                position, time=max(0, self.timestep - self.release_timestep))]).astype(np.float32)

            # Edge information
            edges = []
            for direction in [(0, -35), (35, 0), (0, 35), (-35, 0)]:  # Up, Down, Left, Right
                target_pos = (position[0] + direction[0], position[1] + direction[1])
                target_index = self.state_labels.index(target_pos) if target_pos in self.state_labels else None
                if target_index is not None and self.grid[current_index, target_index] == 1:
                    edges.append(1)
                else:
                    edges.append(0)
            observations[f'{agent_id}_edge_info'] = np.array(edges).astype(np.int8)

        return observations

    def step(self, actions):
        rewards = {}
        terminated = {}
        truncated = {}
        info = {}

        # Process actions for each agent
        for id, action in enumerate(actions):
            agent_id = f'agent_{id}'
            current_pos = self.agent_positions[agent_id]
            current_index = self.state_labels.index(current_pos)
            target_pos = self.get_target_index(current_pos, action)

            # Validate move
            target_index = self.state_labels.index(target_pos) if target_pos in self.state_labels else None
            if target_index is not None and self.grid[current_index, target_index] == 1:
                self.agent_positions[agent_id] = target_pos
            else:
                target_index = current_index  # Invalid move, stay in place

            # Calculate reward
            plume_concentration = self.get_plume_value(
                self.agent_positions[agent_id], time=max(0, self.timestep - self.release_timestep)
            )
            if self.timestep >= self.release_timestep:
                rewards[agent_id] = plume_concentration / 10 - 0.1/self.num_agents
            else:
                rewards[agent_id] = 0.0

            # Termination and truncation
            terminated[agent_id] = plume_concentration > 0.01
            truncated[agent_id] = self.timestep >= self.max_timesteps - 1

        # Aggregate information
        self.timestep += 1
        self.global_timestep += 1
        self.ep_rew += sum(rewards.values()) + 5 if any(terminated.values()) else 0
        rew = sum(rewards.values()) + 5 if any(terminated.values()) else 0


        return self._get_obs(), rew , all(truncated.values()), any(terminated.values()), info

    def get_target_index(self, current_pos, action):
        action_mapping = {
            0: (-35, 0),  # Up
            1: (35, 0),  # Down
            2: (0, -35),  # Left
            3: (0, 35),  # Right
            #4: (0, 0)  # Stay
        }
        x, y = current_pos
        dx, dy = action_mapping[action]
        return (x + dx, y + dy)

    def get_plume_value(self, position, time):
        x, y = position
        # print(time)
        if time == 0:
            return 0.0
        return self.plume_data.get((int(x), int(y), time), 0.0)

    def render(self, mode='human'):
        """
        Render the environment's grid with agent positions, plume concentrations, and edges.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        G = nx.Graph()
        positions = {i: (label[0], label[1]) for i, label in enumerate(self.state_labels)}

        # Add nodes and edges
        for i, label in enumerate(self.state_labels):
            G.add_node(i, pos=positions[i])
            for j in range(len(self.grid[i])):
                if self.grid[i, j] == 1:
                    G.add_edge(i, j)

        # Draw grid edges
        nx.draw_networkx_edges(G, pos=positions, ax=ax, edge_color='gray', alpha=0.5)

        # Plot plume concentrations as heatmap
        # plume_values = [self.plume_data.get((x, y, max(0, self.timestep - self.release_timestep)), 0.0) for x, y in positions.values()]

        plume_values = [self.get_plume_value((x, y), max(0, self.timestep - self.release_timestep)) for x, y in positions.values()]
        scatter = ax.scatter(
            [x for x, y in positions.values()],
            [y for x, y in positions.values()],
            c=plume_values,
            cmap='viridis',
            s=200,
            edgecolors='k'
        )

        # Color bar
        plt.colorbar(scatter, ax=ax, label='Plume Concentration')

        # Plot agents
        for agent_id, position in self.agent_positions.items():
            agent_idx = self.state_labels.index(position)
            agent_x, agent_y = positions[agent_idx]
            ax.scatter(agent_x, agent_y, color='red', s=300, label=f'Agent {agent_id}')

        # Titles and labels
        ax.set_title(f'Timestep: {self.timestep}')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    env = PatrollingEnv('edge_matrix.csv', 'sims', num_agents=3, max_timesteps=200, render_mode="rgb_array")
    env.reset()
    for _ in range(10):
        # actions = {f'agent_{i}': np.random.randint(0, 5) for i in range(3)}
        actions = env.action_space.sample()
        print(actions)
        obs, rewards, terminated, truncated, info = env.step(actions)
        env.render()
        if terminated or truncated:
            break

