import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 100  # Number of time steps
mu = 0.1  # Drift or mean of the random walk (optional)
sigma = 0.2  # Standard deviation of the random shocks

# Generate random shocks
shocks = np.random.normal(0, sigma, T)

# Initialize array to store the random walk values
X = np.zeros(T)
X[0] = 0  # Initial value

# Compute random walk
for t in range(1, T):
    X[t] = X[t-1] + shocks[t]

# Plot the random walk
plt.figure(figsize=(10, 6))
plt.plot(range(T), X, marker='o', linestyle='-', color='b', alpha=0.8)
plt.title('Random Walk Simulation')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Notes
'''
Parameters: T determines the number of time steps. mu and sigma define the mean and standard deviation of the random shocks, respectively.
Random Shocks: np.random.normal(0, sigma, T) generates T random numbers from a normal distribution with mean 0 and standard deviation sigma.
Random Walk Computation: The loop iterates through each time step, updating the value of X based on the previous value and the corresponding random shock.
Visualization: The plot shows the simulated random walk over time.

This basic implementation can be extended by incorporating drift (if desired), simulating multiple paths, or applying the random walk concept to different types of variables or models as needed.
'''
