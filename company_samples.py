import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Parameters for revenue distribution (lognormal)
mean_revenue = 34e9
std_revenue = 69e9
phi_revenue = np.sqrt(std_revenue**2 + mean_revenue**2)
mu_revenue = np.log(mean_revenue**2 / phi_revenue)
sigma_revenue = np.sqrt(np.log(phi_revenue**2 / mean_revenue**2))

# Parameters for y distribution (lognormal)
mean_y = 10e6
std_y = 10e6
phi_y = np.sqrt(std_y**2 + mean_y**2)
mu_y = np.log(mean_y**2 / phi_y)
sigma_y = np.sqrt(np.log(phi_y**2 / mean_y**2))

# Generate samples
samples = []
for i in range(10):
    revenue = np.random.lognormal(mean=mu_revenue, sigma=sigma_revenue)
    y = np.random.lognormal(mean=mu_y, sigma=sigma_y)
    g = max(1.5, np.random.normal(loc=2, scale=1))
    name = f"company{i+1}"
    samples.append(("company", name, revenue, y, g))

# Convert to DataFrame for display
df_samples = pd.DataFrame(samples, columns=["type", "name", "eta", "y", "g"])
df_samples.to_csv("sampled_agents/company_sampled.csv")