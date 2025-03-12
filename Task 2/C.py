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

def monte_carlo_integration(a, b, c, N):
    theta_samples = np.random.uniform(0, np.pi, N)
    return np.mean(f(theta_samples, a, b, c)) * np.pi

# Monte Carlo Error Analysis
beta, c = 0.5, 1  # Given 2beta = c = 1
N_values = [10, 100, 1000, 10000, 100000]
errors_mc = []
exact_value = exact_surface_area(beta, c)

for N in N_values:
    mc_approx = monte_carlo_integration(beta, beta, c, N)
    error = abs(mc_approx - exact_value) / exact_value
    errors_mc.append(error)

# Plot Monte Carlo Errors
plt.figure(figsize=(8, 6))
plt.loglog(N_values, errors_mc, marker='o', linestyle='-', label='Monte Carlo Error')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Relative Error')
plt.title('Monte Carlo Error Convergence')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()