
import numpy as np
import pandas as pd

data = pd.read_csv("rgb_signals.csv")

time = data["Time"].to_numpy(dtype=float)
dt = np.diff(time)

print("\n========== DETAILED GAP ANALYSIS ==========")

print("Time interval percentiles:")

for p in [10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"{p}th percentile: {np.percentile(dt, p):.5f} seconds")

print("\nGap counts:")

for threshold in [0.04, 0.05, 0.075, 0.10, 0.15, 0.20]:
    count = np.sum(dt > threshold)
    print(f"Gaps above {threshold:.3f} seconds: {count}")

# Effective average sampling frequency
fs_average = (len(time) - 1) / (time[-1] - time[0])

print(f"\nEffective average sampling rate: {fs_average:.2f} Hz")

print("\n===========================================")
