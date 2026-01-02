import pandas as pda
import numpy as npy
import matplotlib.pyplot as mat

# Loading the dataset
df = pda.read_csv("rawData.csv")
# Plot 1: Raw Data
mat.figure(figsize=(6, 6))
mat.scatter(df['x'], df['y'], s=5)
mat.title("Plot 1: Raw Data (x vs y)")
mat.xlabel("x")
mat.ylabel("y")
mat.show()


# Remove deceptive points

x_low, x_high = npy.percentile(df['x'], [5, 95])
y_low, y_high = npy.percentile(df['y'], [5, 95])

filtered = df[
    (df['x'] >= x_low) & (df['x'] <= x_high) &
    (df['y'] >= y_low) & (df['y'] <= y_high)
]


# Normalize manually (Min-Max)
x_norm = (filtered['x'] - filtered['x'].min()) / (filtered['x'].max() - filtered['x'].min())
y_norm = (filtered['y'] - filtered['y'].min()) / (filtered['y'].max() - filtered['y'].min())

# Plot 2: Final Result
mat.figure(figsize=(6, 6))
mat.scatter(x_norm, y_norm, s=5)
mat.title("Plot 2: Filtered & Normalized Data")
mat.xlabel("x (normalized)")
mat.ylabel("y (normalized)")
mat.show()
