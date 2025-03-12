import numpy as np
import matplotlib.pyplot as plt

def sobol_2d(n, m1=1, m2=3, m3=5, v1=1/2, v2=3/4, v3=5/8):
    # Direction numbers
    direction_numbers = np.array([v1, v2, v3], dtype=float)
    m = np.array([m1, m2, m3], dtype=int)
    
    # Compute the first 50 Sobol sequence elements
    sobol_points = np.zeros((n, 2))
    x = np.zeros((n, 2), dtype=float)
    
    for i in range(1, n):
        lsb = (i) & -(i)  # Get the least significant bit
        index = int(np.log2(lsb))
        
        if index < len(direction_numbers):
            x[i, 0] = x[i - 1, 0] + direction_numbers[index] % 1  # Mod 1 ensures values stay in [0,1]
            x[i, 1] = x[i - 1, 1] + (m[index] / (1 << (index + 1))) % 1
        else:
            x[i] = x[i - 1]
        
        sobol_points[i] = x[i]
    
    return sobol_points

# Generate and plot the first 50 elements
n_samples = 50
sobol_data = sobol_2d(n_samples)

plt.figure(figsize=(6, 6))
plt.scatter(sobol_data[:, 0], sobol_data[:, 1], color='blue')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D Sobol Sequence')
plt.grid(True)
plt.show()
