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

def monte_carlo_integral(a, b, c, N, proposal='uniform'):
    if proposal == 'uniform':
        theta_samples = np.random.uniform(0, np.pi, N)
        weights = np.ones(N)
    elif proposal == 'exp':
        u_samples = np.random.uniform(0, 1, N)
        theta_samples = -np.log(1 - u_samples) / 3  # Inverse transform for q1(x) = exp(-3x)
        weights = np.exp(3 * theta_samples)  # Importance weight
    elif proposal == 'sin2':
        u_samples = np.random.uniform(0, 1, N)
        theta_samples = (1 / 5) * np.arcsin(u_samples)  # Inverse transform for q2(x) = sin^2(5x)
        weights = 1 / (5 * np.sin(5 * theta_samples)**2)  # Importance weight
    else:
        raise ValueError("Unknown proposal function")
    
    return np.pi * np.mean(f(theta_samples, a, b, c) * weights)

# Monte Carlo error analysis
beta, c = 0.5, 0.5  # Fixed values for Monte Carlo test
N_samples = [10, 100, 1000, 10000, 100000]
exact_value = exact_surface_area(beta, c)

mc_errors = {"uniform": [], "exp": [], "sin2": []}
for N in N_samples:
    for method in mc_errors.keys():
        mc_approx = monte_carlo_integral(beta, beta, c, N, proposal=method)
        mc_errors[method].append(abs(mc_approx - exact_value) / exact_value)

# Plot Monte Carlo error
plt.figure(figsize=(8, 6))
for method, errors in mc_errors.items():
    plt.plot(N_samples, errors, marker='o', linestyle='-', label=f'MC Error ({method})')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Relative Error')
plt.title('Monte Carlo Error Analysis with Different Sampling Methods')
plt.legend()
plt.grid()
plt.show()
