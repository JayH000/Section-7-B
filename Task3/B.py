import numpy as np
import matplotlib.pyplot as plt

def rejection_sampling_exponential(target_pdf, M, N):
    """
    Perform rejection sampling using an exponential proposal distribution q(t) = 2e^(-2t).
    
    Parameters:
    - target_pdf: function, the target probability density function p(t)
    - M: float, scaling constant ensuring M * q(t) ≥ p(t) for all t
    - N: int, number of accepted samples
    
    Returns:
    - samples: numpy array of shape (N,)
    - rejection_ratio: float, ratio of accepted samples to total proposals
    """
    samples = []
    num_attempts = 0  # Total proposals made

    while len(samples) < N:
        u1, u2 = np.random.uniform(0, 1, 2)  # Generate two uniform samples
        t_star = -np.log(u1) / 2  # Sample from Exp(1) using inverse transform

        num_attempts += 1  # Count every sample attempt

        if u2 < target_pdf(t_star) / (M * 2 * np.exp(-2 * t_star)):  # Accept condition
            samples.append(t_star)

    rejection_ratio = N / num_attempts  # Accepted / Total proposals
    return np.array(samples), rejection_ratio

# Define target PDF (same as before)
def target_pdf(t):
    return np.exp(-t)  # Exponential decay

# Choose M (must satisfy M * q(t) ≥ p(t))
M = 1.5

# Sample sizes
N_values = [100, 1000, 10000]
hist_bins = 30

# Plot results for different N
plt.figure(figsize=(12, 6))

for i, N in enumerate(N_values):
    samples_exp, rejection_ratio_exp = rejection_sampling_exponential(target_pdf, M, N)
    
    plt.subplot(1, 3, i + 1)
    plt.hist(samples_exp, bins=hist_bins, density=True, alpha=0.6, label=f'N={N}')
    t_vals = np.linspace(0, 5, 100)
    plt.plot(t_vals, target_pdf(t_vals), 'r-', label='Target PDF')
    plt.title(f'Exp Proposal N={N}, Rejection Ratio={rejection_ratio_exp:.2f}')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()
