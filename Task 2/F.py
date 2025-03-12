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

def monte_carlo_uniform(a, b, c, N):
    theta_samples = np.random.uniform(0, np.pi, N)
    return np.mean(f(theta_samples, a, b, c)) * np.pi

def monte_carlo_gaussian(a, b, c, N, mu=0, sigma=1):
    theta_samples = np.random.normal(mu, sigma, N)
    theta_samples = np.clip(theta_samples, 0, np.pi)  # Restrict to valid range
    return np.mean(f(theta_samples, a, b, c)) * np.pi

# Monte Carlo Error Analysis
beta, c = 0.5, 1  # Given 2beta = c = 1
N_values = [10, 100, 1000, 10000, 100000]
errors_mc_uniform = []
errors_mc_gaussian = []
exact_value = exact_surface_area(beta, c)

for N in N_values:
    mc_uniform = monte_carlo_uniform(beta, beta, c, N)
    mc_gaussian = monte_carlo_gaussian(beta, beta, c, N)
    
    error_uniform = abs(mc_uniform - exact_value) / exact_value
    error_gaussian = abs(mc_gaussian - exact_value) / exact_value
    
    errors_mc_uniform.append(error_uniform)
    errors_mc_gaussian.append(error_gaussian)

# Plot Monte Carlo Errors
plt.figure(figsize=(8, 6))
plt.loglog(N_values, errors_mc_uniform, marker='o', linestyle='-', label='Uniform MC Error')
plt.loglog(N_values, errors_mc_gaussian, marker='s', linestyle='-', label='Gaussian MC Error')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Relative Error')
plt.title('Monte Carlo Error Convergence (Uniform vs Gaussian)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Test various µ and σ with fixed N=10000
mu_values = [-1, 0, 1]
sigma_values = [0.5, 1, 2]
errors_gaussian_variation = np.zeros((len(mu_values), len(sigma_values)))

for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        mc_gaussian = monte_carlo_gaussian(beta, beta, c, 10000, mu, sigma)
        errors_gaussian_variation[i, j] = abs(mc_gaussian - exact_value) / exact_value

# Heatmap for different µ and σ
plt.figure(figsize=(8, 6))
plt.imshow(errors_gaussian_variation, extent=[0.5, 2, -1, 1], origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='Relative Error')
plt.xlabel('Sigma (σ)')
plt.ylabel('Mu (µ)')
plt.title('Error Heatmap for Different Gaussian Distributions')
plt.show()