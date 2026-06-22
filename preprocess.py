import pandas as pd
import numpy as np

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

df = df.sort_values("Order Date")
df = df.set_index("Order Date")

# -----------------------
# Aggregate to daily sales
# -----------------------
daily_sales = df["Sales"].resample("D").sum()

# -----------------------
# Fix zeros (important for log)
# -----------------------
daily_sales = daily_sales + 1

# -----------------------
# Log transform (stabilizes variance)
# -----------------------
daily_sales_log = np.log1p(daily_sales)

# -----------------------
# Train/test split (LOG SCALE)
# -----------------------
train = daily_sales_log[:-30]
test = daily_sales_log[-30:]

# -----------------------
# Save outputs for next file
# -----------------------
train.to_csv("train_series.csv")
test.to_csv("test_series.csv")
daily_sales.to_csv("daily_sales.csv")

print("Preprocessing complete.")
print("Files saved: train_series.csv, test_series.csv, daily_sales.csv")