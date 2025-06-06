import numpy as np
from scipy.stats import norm, lognorm
from scipy.special import erfinv
import pandas as pd

def lognormal_from_mean_gini(mean, gini):
    """
    Return a frozen SciPy log‑normal distribution whose arithmetic mean
    and Gini coefficient match the inputs.

    Parameters
    ----------
    mean : float
        Desired arithmetic mean (μₓ) of the distribution, must be > 0.
    gini : float
        Desired Gini coefficient, 0 < gini < 1.

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        A frozen log‑normal distribution object. Use .rvs(size) to sample.
    """
    if not (0 < gini < 1):
        raise ValueError("gini must be between 0 and 1 (exclusive).")
    if mean <= 0:
        raise ValueError("mean must be positive.")


    sigma = np.sqrt(2) * norm.ppf((gini + 1) / 2.0)

    
    mu = np.log(mean) - 0.5 * sigma**2

    # 3. Build and return a frozen distribution.
    return lognorm(s=sigma, scale=np.exp(mu)).ppf
def gini_from_grouped_data(pop_shares, emission_shares):
    """
    Compute Gini coefficient from grouped data using the trapezoidal approximation:
    G = 1 - Σ (Y_i + Y_{i-1}) * (X_i - X_{i-1})
    
    Parameters:
    - pop_shares: list of cumulative population shares (X)
    - emission_shares: list of cumulative emissions shares (Y)
    
    Both lists must start with 0 and end with 1.
    """
    assert len(pop_shares) == len(emission_shares), "Lists must be the same length"
    
    gini = 1
    for i in range(1, len(pop_shares)):
        x_diff = pop_shares[i] - pop_shares[i - 1]
        y_sum = emission_shares[i] + emission_shares[i - 1]
        gini -= y_sum * x_diff
    return gini

# Cumulative population shares (X) and cumulative emission shares (Y)
pop_shares = [0.0, 0.5, 0.9, 1.0]
emission_shares = [0.0, 0.12, 0.52, 1.0]




if __name__ == "__main__":
    mean_income = 23380      # arithmetic mean
    gini_coefficient_income = 0.67  # Gini
    mean_carbon = 4.7 #cited
    gini_coefficient_carbon = gini_from_grouped_data(pop_shares, emission_shares)


    pop_shares = [0.0, 0.5, 0.9, 1.0]
    emission_shares = [0.0, 0.12, 0.52, 1.0]

    income_ppf = lognormal_from_mean_gini(mean_income, gini_coefficient_income)
    carbon_ppf = lognormal_from_mean_gini(mean_carbon, gini_coefficient_carbon)


    percentiles = np.random.uniform(0, 1, 1000)

# Get correlated samples by evaluating both PPFs at the same percentiles
    samples = [(carbon_ppf(p), income_ppf(p)) for p in percentiles]
    person_samples = [
    ("person", f"person{i+1}", income, carbon, 3)
    for i, (carbon, income) in enumerate(samples)
    ]

# Create DataFrame
    df_people = pd.DataFrame(person_samples, columns=["type", "name", "eta", "y", "g"])
        # for c, i in samples:
        # print(f"Carbon: {c:.2f}, Income: {i:.2f}")
    df_people.to_csv("people_sampled.csv")
