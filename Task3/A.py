import numpy as np
import matplotlib.pyplot as plt

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
    num_attempts = 0  # Total proposals made
    
    while len(samples) < N:
        t_star = np.random.uniform(0, t_f)  # Sample from U(0, t_f)
        u = np.random.uniform(0, 1)  # Sample from U(0,1)
        num_attempts += 1  # Count every sample attempt
        
        if u < target_pdf(t_star) / (M / t_f):  # Accept condition
            samples.append(t_star)
    
    rejection_ratio = N / num_attempts  # Accepted / Total proposals
    return np.array(samples), rejection_ratio

# Define target PDF (e.g., exponential distribution)
def target_pdf(t):
    return np.exp(-t)  # Exponential decay

# Choosing t_f and M
t_f = 5  # Upper bound for uniform proposal
M = 1.5  # Scaling factor, must satisfy M * q(t) ≥ p(t)

# Sample sizes
N_values = [100, 1000, 10000]
hist_bins = 30

# Plot results for different N
plt.figure(figsize=(12, 6))

for i, N in enumerate(N_values):
    samples, rejection_ratio = rejection_sampling(target_pdf, t_f, M, N)
    
    plt.subplot(1, 3, i + 1)
    plt.hist(samples, bins=hist_bins, density=True, alpha=0.6, label=f'N={N}')
    t_vals = np.linspace(0, t_f, 100)
    plt.plot(t_vals, target_pdf(t_vals), 'r-', label='Target PDF')
    plt.title(f'N={N}, Rejection Ratio={rejection_ratio:.2f}')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()