import numpy as np
import matplotlib.pyplot as plt

def box_muller_gaussian(n_samples, mu=0, sigma=1):
    """Generate Gaussian-distributed samples using the Box-Muller transform."""
    U1 = np.random.rand(n_samples // 2)
    U2 = np.random.rand(n_samples // 2)
    
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)

    # Scale and shift to match desired mean and standard deviation
    X0 = mu + sigma * Z0
    X1 = mu + sigma * Z1

    samples = np.concatenate((X0, X1))

    # If n_samples is odd, generate one extra sample
    if n_samples % 2 != 0:
        U1_extra, U2_extra = np.random.rand(), np.random.rand()
        Z_extra = np.sqrt(-2 * np.log(U1_extra)) * np.cos(2 * np.pi * U2_extra)
        samples = np.append(samples, mu + sigma * Z_extra)

    return samples

# Parameters
mu = 3    # Mean
sigma = 2  # Standard deviation
N_samples = [10, 100, 1000, 10000, 100000]

# Generate samples and store them
sample_data = {}

plt.figure(figsize=(12, 8))
for i, N in enumerate(N_samples):
    samples = box_muller_gaussian(N, mu, sigma)
    sample_data[N] = samples  # Store samples in a dictionary

    plt.subplot(2, 3, i + 1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='b', label=f"N={N}")

    # Overlay theoretical normal distribution
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    plt.plot(x, pdf, 'r-', label="Theoretical PDF")

    plt.legend()
    plt.title(f"Histogram (N={N})")
    plt.xlabel("Value")
    plt.ylabel("Density")

plt.tight_layout()
plt.show()

# Convert sample_data to numpy array
sample_array = {N: np.array(samples) for N, samples in sample_data.items()}
