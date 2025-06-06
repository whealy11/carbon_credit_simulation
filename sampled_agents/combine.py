import pandas as pd

# Replace with your actual file paths
file1 = "sampled_agents/company_sampled.csv"
file2 = "sampled_agents/country_sampled.csv"
file3 = "sampled_agents/people_sampled.csv"

# Read the CSVs
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Combine them into one DataFrame
df = pd.concat([df1, df2, df3], ignore_index=True)

# Optional: Save to a new CSV
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Ensure correct column order and types
df.columns = ["type", "name", "revenue", "y", "g"]

# Reformat and sort if needed
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["g"] = pd.to_numeric(df["g"], errors="coerce")

# Optional: Reset index and sort by type/name
df = df.reset_index(drop=True).sort_values(by=["type", "name"])

# Save the cleaned file
df.to_csv("combined.csv", index=False)
