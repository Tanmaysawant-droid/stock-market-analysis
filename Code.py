import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy import stats

# Download the stock info/data:
data = yf.download("RELIANCE.NS", start="2020-01-01", end="2024-01-01")
data = data.dropna()

print(data.head())
print(data.info())

# Plotting Close, 20 and 50 days average and comparing them:
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

print(data[["Close", "MA20", "MA50"]].tail())

plt.figure(figsize=(12,6))

plt.plot(data["Close"], label="Close Price", linewidth=2)
plt.plot(data["MA20"], label="20-Day MA", linestyle="--", color="orange")
plt.plot(data["MA50"], label="50-Day MA", linestyle="--", color="blue")

plt.title("Reliance Stock Price with Moving Averages", fontsize=20)
plt.xlabel("Date", size=15)
plt.ylabel("Price (INR)", size=15)
plt.legend()
plt.grid(True)
plt.show() 

# Daily Returns:
data["Daily Return"] = data["Close"].pct_change()

returns = data["Daily Return"].dropna()

mean_return = np.mean(returns)
std_return = np.std(returns)

print("Sharp-like ratio:", float(mean_return / std_return))
print("Mean Daily Return:", float(mean_return))
print("Volatility (Std Dev):", float(std_return))

plt.figure(figsize=(10,5))

plt.plot(data["Daily Return"], linewidth=1.5)
plt.title("Daily Returns of Reliance Stock", size=20)
plt.xlabel("Date", size=15)
plt.ylabel("Daily Return(%)", size=15)
plt.legend(["Daily Returns"])
plt.grid(alpha=0.3)
plt.show() 

# Trend analysis using regression
df = data.copy()
df = df.dropna()

x = np.arange(len(df))

y = df["Close"].values.flatten()

print("Data points:", len(df))

slope, intercept, r_value, p_value, std_error = stats.linregress(x, y)

trend_line = slope * x + intercept

print("Trend slope:", slope)
print("R-squared:", r_value**2)

# Trendplot
plt.figure(figsize=(12,6))

plt.plot(x, y, label="Close Price")
plt.plot(x, trend_line, label="Trend Line", linestyle="--")
plt.title("Stock Trend Analysis (Regression)", size = 20)
plt.xlabel("Date", size = 15)
plt.ylabel("Closing Price (INR)", size = 15)
plt.legend()
plt.grid(alpha = 0.3)
plt.show()