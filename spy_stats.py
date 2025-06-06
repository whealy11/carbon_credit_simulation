import yfinance as yf
import pandas as pd
import numpy as np

# Get list of S&P 500 tickers
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = sp500['Symbol'].tolist()

revenues = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        if fin is not None and 'Total Revenue' in fin.index:
            revenue = fin.loc['Total Revenue'].iloc[0]
            revenues.append(revenue)
    except:
        continue

# Convert to numpy array and compute variance
revenues = np.array(revenues, dtype=np.float64)
revenues = revenues[~np.isnan(revenues)]

# Compute variance
variance = np.var(revenues)
print(np.sqrt(variance))
print(f"Variance of S&P 500 company revenues: {variance:.2e}")
print(f"STD of S&P 500 company revenues: {np.sqrt(variance):.2e}")
print("average revenue: ", np.mean(revenues))
