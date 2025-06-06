import pandas as pd
df = pd.read_csv("Country_GDP_and_Emissions.csv")

# Randomly sample 3 rows
sampled_countries = df.sample(n=3, random_state=42)

# Format into new DataFrame
country_df = pd.DataFrame({
    "type": "country",
    "name": sampled_countries["Country"],
    "eta": sampled_countries["GDP_2025_MillionUSD"] * 1e6,
    "y": sampled_countries["CO2_Emissions_2022_Tons"],
    "g": 1.5
})

# Save to CSV
output_path = "sampled_agents/country_sampled.csv"
country_df.to_csv(output_path, index=False)

