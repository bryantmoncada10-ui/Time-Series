import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 1. Load data
# -----------------------
df = pd.read_csv("train.csv")

# -----------------------
# 2. Convert date column
# -----------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True) 
df['Order Date'] = pd.to_datetime(df['Order Date'])

# -----------------------
# 3. Sort + set index (time series setup)
# -----------------------
df = df.sort_values('Order Date')
df = df.set_index('Order Date')

# -----------------------
# 4. Create time series (AGGREGATED SALES)
# -----------------------
monthly_sales = df['Sales'].resample('ME').sum()

# -----------------------
# 5. Quick check
# -----------------------
print(monthly_sales.head())

# -----------------------
# 6. Plot time series
# -----------------------
monthly_sales.plot(figsize=(12, 5))
plt.title("Monthly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()