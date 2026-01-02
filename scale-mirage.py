import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("rawData.csv")

# -------------------------
# Plot 1: Raw Data
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(df['x'], df['y'], s=5)
plt.title("Plot 1: Raw Data (x vs y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------
# Remove deceptive points
# -------------------------
x_low, x_high = np.percentile(df['x'], [5, 95])
y_low, y_high = np.percentile(df['y'], [5, 95])

filtered = df[
    (df['x'] >= x_low) & (df['x'] <= x_high) &
    (df['y'] >= y_low) & (df['y'] <= y_high)
]

# -------------------------
# Normalize manually (Min-Max)
# -------------------------
x_norm = (filtered['x'] - filtered['x'].min()) / (filtered['x'].max() - filtered['x'].min())
y_norm = (filtered['y'] - filtered['y'].min()) / (filtered['y'].max() - filtered['y'].min())

# -------------------------
# Plot 2: Final Result
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(x_norm, y_norm, s=5)
plt.title("Plot 2: Filtered & Normalized Data")
plt.xlabel("x (normalized)")
plt.ylabel("y (normalized)")
plt.show()