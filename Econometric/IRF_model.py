import numpy as np
import matplotlib.pyplot as plt
# Define system parameters
T = 2.0
K = 3.0
# Define time values
t = np.linspace(0, 10, 1000)
# Calculate impulse response
h = K * (1 / T) * np.exp(-t / T)
# Plot the impulse response
plt.figure(figsize=(8, 6))
plt.plot(t, h, label='Impulse Response')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Impulse Response of a First-Order System')
plt.grid(True)
plt.legend()
plt.show()


