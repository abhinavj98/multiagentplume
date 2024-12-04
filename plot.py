import matplotlib.pyplot as plt

# Data for number of agents and mean timesteps
agents = [1, 2, 3, 4]
mean_timesteps = [23.10, 49.06, 74.29, 79.20]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(agents, mean_timesteps, marker='o', linestyle='-', color='b', label='Mean Timesteps')

# Adding labels and title
plt.title('Mean Hitting TimeSteps vs Number of Agents', fontsize=14)
plt.xlabel('Number of Agents', fontsize=12)
plt.ylabel('Mean Hitting timesteps', fontsize=12)

# Show grid for better readability
plt.grid(True)

# Show the plot
plt.legend()
plt.show()
