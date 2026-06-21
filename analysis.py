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

# Sort + set index (time series setup)
df = df.sort_values('Order Date')
df = df.set_index('Order Date')

# Create time series (AGGREGATED SALES)
daily_sales = df['Sales'].resample('D').sum()

# Quick check
print(daily_sales.head())

# Plot time series
rolling_7 = daily_sales.rolling(window=7).mean()

plt.figure(figsize=(12,5))
plt.plot(daily_sales, alpha=0.4, label='Daily Sales')
plt.plot(rolling_7, linewidth=2, label='7-Day Average')
plt.title('Daily Sales with 7-Day Rolling Average')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.savefig("daily_sales_rolling.png")
print("Plot saved successfully")
plt.show()



# Checking for trends, holiday sales, and seasonality

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(
    daily_sales,
    model='additive',
    period=7
)

decomposition.plot()
plt.show()




#Test if the data is stationary before using ARIMA
# ADF Test

from statsmodels.tsa.stattools import adfuller
result = adfuller(daily_sales)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] <= 0.05:
    print("Series is stationary")
else:
    print("Series is NOT stationary")



# TRAIN/TEST SPLIT
train = daily_sales[:-30]
test = daily_sales[-30:]

print("Train size:", len(train))
print("Test size:", len(test))

# Visualize the split
plt.figure(figsize=(12,5))

plt.plot(train, label='Train')
plt.plot(test, label='Test')

plt.title('Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.legend()
plt.show()

# Train the model with train splot
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 7)
)

results = model.fit()

print(results.summary())

# Predict with Test split

predictions = results.predict(
    start=test.index[0],
    end=test.index[-1]
)

print(predictions.head())

# Plot prediction vs actual

plt.figure(figsize=(12,5))

plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(predictions, label='Predicted')

plt.title('SARIMA Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.legend()
plt.show()

#Calculate MAE
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test, predictions)

print("MAE:", mae)