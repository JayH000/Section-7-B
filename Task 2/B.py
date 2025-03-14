import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

def exact_surface_area(beta, c):
    e = np.sqrt(1 - (beta**2 / c**2))
    a = max(beta, c)  # Assume semi-major axis is the max
    return 2 * np.pi * beta**2 * (1 + (c / a) * np.arcsin(e))

def f(theta, a, b, c):
    return 2 * np.pi * c * np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2)

def midpoint_rule(a, b, c, N=1000):
    theta_vals = np.linspace(0, np.pi, N+1)
    midpoints = (theta_vals[:-1] + theta_vals[1:]) / 2
    return np.sum(f(midpoints, a, b, c)) * (np.pi / N)

def gaussian_quadrature(a, b, c, N=10):
    nodes, weights = roots_legendre(N)
    nodes = 0.5 * (nodes + 1) * np.pi  # Transform from [-1,1] to [0,pi]
    weights *= np.pi / 2
    return np.sum(weights * f(nodes, a, b, c))

# Grid of beta and c values
beta_vals = np.logspace(-3, 3, 50)
c_vals = np.logspace(-3, 3, 50)
errors = np.zeros((50, 50))

for i, beta in enumerate(beta_vals):
    for j, c in enumerate(c_vals):
        exact = exact_surface_area(beta, c)
        approx = gaussian_quadrature(beta, beta, c)  # Using Gaussian quadrature
        errors[i, j] = abs(approx - exact) / exact

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(errors, extent=[-3, 3, -3, 3], origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='Relative Error')
plt.xlabel('log10(beta)')
plt.ylabel('log10(c)')
plt.title('Error Heatmap of Surface Area Approximation')
plt.show()
