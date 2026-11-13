
import numpy as np
import pandas as pd

# Load original RGB measurements
data = pd.read_csv("rgb_signals.csv")

time = data["Time"].to_numpy(dtype=float)

# Identify unusually large gaps
MAX_GAP = 0.10

dt = np.diff(time)

gap_indices = np.where(dt > MAX_GAP)[0]

# Divide recording into continuous segments
starts = np.concatenate(([0], gap_indices + 1))
ends = np.concatenate((gap_indices, [len(time) - 1]))

print("\n========== RECORDING SEGMENTS ==========")

segments = []

for i, (start, end) in enumerate(zip(starts, ends)):

    duration = time[end] - time[start]
    samples = end - start + 1

    segments.append({
        "Segment": i + 1,
        "Samples": samples,
        "Duration": duration,
        "Start": time[start],
        "End": time[end]
    })

    print(
        f"Segment {i + 1}: "
        f"{samples} samples | "
        f"{duration:.2f} seconds | "
        f"Start: {time[start]:.2f}s | "
        f"End: {time[end]:.2f}s"
    )

print("\n========================================")

# Identify longest segment
longest = max(segments, key=lambda x: x["Duration"])

print("\nLongest continuous segment:")
print(f"Duration: {longest['Duration']:.2f} seconds")
print(f"Samples: {longest['Samples']}")

