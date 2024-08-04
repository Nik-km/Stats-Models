import numpy as np
import matplotlib.pyplot as plt

def gen_impulse(n, shock):
    p = 0.9
    z_ss = 1
    k_ss = 9.354
    beta = 2.16
    delta = 0.017
    
    z = np.zeros(n)
    k = np.zeros(n)
    
    # Initial conditions
    z[0] = z_ss
    k[0] = k_ss
    
    for t in range(1, n):
        z[t] = p * z[t-1] + (1 - p) * z_ss + shock * (t == 0)
        k[t] = k[t-1] * (1 - delta) + beta * (z[t-1] - 1) + delta * (k[t-1] - k_ss) + k_ss
    
    return k, z

def plot_impulses(k, z):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(k, label='Capital (k)')
    plt.title('Impulse Response: Capital')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(z, label='Z')
    plt.title('Impulse Response: Z')
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Generate impulse response for capital and z variables
n_periods = 40
shock = 0.01
capital_impulse, z_impulse = gen_impulse(n_periods, shock)
plot_impulses(capital_impulse, z_impulse)

def generate_labor_output(k, z):
    labor = np.zeros_like(k)
    output = np.zeros_like(k)
    
    for t in range(len(k)):
        labor[t] = 0.33 + 0.809 * (z[t] - 1) + 0.025 * (k[t] - 9.354)
        output[t] = z[t] ** labor[t] * k[t] ** 0.33
    
    return labor, output

def plot_labor_output(labor, output):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(labor, label='Labor (n)')
    plt.title('Impulse Response: Labor')
    plt.xlabel('Time')
    plt.ylabel('Labor')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(output, label='Output (y)')
    plt.title('Impulse Response: Output')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Generate labor and output impulse responses
labor_impulse, output_impulse = generate_labor_output(capital_impulse, z_impulse)
plot_labor_output(labor_impulse, output_impulse)
