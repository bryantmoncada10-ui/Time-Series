import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------------------------
# 1. Load Original Daily Sales (Before train/test log split)
# -------------------------------------------------------------------------
print("Loading daily sales data...")
# Loading the un-split series to create proper historical lag features
daily_sales = pd.read_csv("daily_sales.csv", index_col=0, parse_dates=True).squeeze()
daily_sales = daily_sales.asfreq('D')

# -------------------------------------------------------------------------
# 2. Feature Engineering (The Secret Sauce for ML Time Series)
# -------------------------------------------------------------------------
df_features = pd.DataFrame(index=daily_sales.index)
df_features['target'] = daily_sales.values

# Calendar Features
df_features['dayofweek'] = df_features.index.dayofweek
df_features['day'] = df_features.index.day
df_features['month'] = df_features.index.month
df_features['is_weekend'] = df_features.index.dayofweek.isin([5, 6]).astype(int)

# Lag Features (Looking back in time)
df_features['lag_1'] = daily_sales.shift(1)
df_features['lag_2'] = daily_sales.shift(2)
df_features['lag_7'] = daily_sales.shift(7) # Crucial for weekly spikes

# Rolling Features (Capturing recent trends)
df_features['rolling_mean_7'] = daily_sales.shift(1).rolling(window=7).mean()

# Drop rows with NaN values created by shifting/rolling
df_features = df_features.dropna()

# -------------------------------------------------------------------------
# 3. Train / Test Split (Last 30 days for testing)
# -------------------------------------------------------------------------
X = df_features.drop(columns=['target'])
y = df_features['target']

X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

# -------------------------------------------------------------------------
# 4. Train the Machine Learning Model
# -------------------------------------------------------------------------
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
predictions = pd.Series(predictions, index=y_test.index)

# Adjust back for the "+1" shift done in your original preprocessing script
# (Since we loaded daily_sales which already had +1 applied)
predictions = predictions - 1
test_actual = y_test - 1

# Ensure no negative predictions
predictions = predictions.clip(lower=0)

# -------------------------------------------------------------------------
# 5. Evaluation
# -------------------------------------------------------------------------
mae = mean_absolute_error(test_actual, predictions)
print("\n" + "="*30)
print(f"Random Forest Model Evaluation\nMAE: {mae:.2f}")
print("="*30 + "\n")

# -------------------------------------------------------------------------
# 6. Plot Results
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(test_actual, label="Actual Sales", color="tab:blue", linewidth=2)
plt.plot(predictions, label="Random Forest Forecast", color="tab:orange", linestyle="--", linewidth=2)

plt.title("Actual vs. Random Forest Predicted Daily Sales", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()

# Calculate wMAPE (Safe for zero-sales days)
wmape = (np.sum(np.abs(test_actual - predictions)) / np.sum(test_actual)) * 100
print(f"wMAPE (Real Percentage Error): {wmape:.2f}%")