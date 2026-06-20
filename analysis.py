import pandas as pd

df = pd.read_csv("train.csv")

print(df.head())

# Inspect data set
print(df.shape)
print(df.columns)
print(df.info())

print(df["Order Date"].head())
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Order Date"] = pd.to_datetime(df["Order Date"])

print(df["Order Date"].dtype)
print(df["Order Date"].head())