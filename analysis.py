import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error

# Loading the dataset
df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

df = df.sort_values("Order Date")
df = df.set_index("Order Date")

# Maje daily sales
daily_sales = df["Sales"].resample("D").sum()

# Log transformation for better outcome (stabilize variance, SARIMAX model said variance might be unstable)
daily_sales_log = np.log1p(daily_sales)

# Visualize every day sales along with 7 day rolling mean
rolling_7 = daily_sales.rolling(window=7).mean()

plt.figure(figsize=(12,5))
plt.plot(daily_sales, alpha=0.4, label="Daily Sales")
plt.plot(rolling_7, linewidth=2, label="7-Day Average")
plt.title("Daily Sales + Rolling Mean")
plt.legend()
plt.show()

# EXTRACTING SEASONALITY/TREND/PATTERNS
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(daily_sales, model="additive", period=7)
decomposition.plot()
plt.show()

# CHECK IF STATIONARY
result = adfuller(daily_sales_log)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] <= 0.05:
    print("Series is stationary")
else:
    print("Series is NOT stationary")

# Create TRAIN/TEST SPLIT
train = daily_sales_log[:-30]
test = daily_sales_log[-30:]

print("Train size:", len(train))
print("Test size:", len(test))

plt.figure(figsize=(12,5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.legend()
plt.show()

# SARIMAX MODELING using TRAIN SPLOT
model = SARIMAX(
    train,
    order=(2, 1, 2),              
    seasonal_order=(1, 1, 1, 7),  
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()

print(results.summary())

# Predict on the test set
predictions_log = results.predict(
    start=test.index[0],
    end=test.index[-1]
)

# Comvert back to normal scale
predictions = np.expm1(predictions_log)
test_actual = np.expm1(test)

# Plot prediction vs actual
plt.figure(figsize=(12,5))
plt.plot(daily_sales[-60:], label="Actual (last 60 days)")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("SARIMAX Forecast vs Actual")
plt.show()

# Calculate MAE and MAPE
mae = mean_absolute_error(test_actual, predictions)
print("MAE:", mae)


mape = (np.abs((test_actual - predictions) / test_actual)).mean() * 100
print("MAPE (%):", mape)