import numpy as np

def rejection_sampling(target_pdf, t_f, M, n_samples):
    """
    Perform rejection sampling with a uniform proposal distribution U(0, t_f).
    
    Parameters:
    - target_pdf: function, the target probability density function p(t)
    - t_f: float, upper bound for uniform proposal distribution
    - M: float, scaling constant such that M * q(t) >= p(t) for all t
    - n_samples: int, number of accepted samples to generate
    
    Returns:
    - samples: list of accepted samples
    """
    samples = []
    
    while len(samples) < n_samples:
        t_star = np.random.uniform(0, t_f)  # Sample from U(0, t_f)
        u = np.random.uniform(0, 1)  # Sample from U(0,1)
        
        if u < target_pdf(t_star) / (M / t_f):  # Accept condition
            samples.append(t_star)
    
    return np.array(samples)

# Example usage
def target_pdf(t):
    """ Example target PDF, must be properly bounded for rejection sampling. """
    return np.exp(-t)  # Example: Exponential distribution

t_f = 5  # Upper bound for uniform proposal
M = 1.5  # Scaling factor, must satisfy M * q(t) ≥ p(t)
n_samples = 1000  # Number of samples to generate

samples = rejection_sampling(target_pdf, t_f, M, n_samples)

# Plot the results
import matplotlib.pyplot as plt
plt.hist(samples, bins=30, density=True, alpha=0.6, label='Rejection Sampling')
t_vals = np.linspace(0, t_f, 100)
plt.plot(t_vals, target_pdf(t_vals), 'r-', label='Target PDF')
plt.legend()
plt.show()
