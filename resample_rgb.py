
import numpy as np
import pandas as pd
import os

# ==================================================
# 1. LOAD ORIGINAL RGB SIGNALS
# ==================================================

input_file = "rgb_signals.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError("rgb_signals.csv not found.")

data = pd.read_csv(input_file)

required_columns = [
    "Time",
    "Forehead_R", "Forehead_G", "Forehead_B",
    "LeftCheek_R", "LeftCheek_G", "LeftCheek_B",
    "RightCheek_R", "RightCheek_G", "RightCheek_B"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")

data = data[required_columns].copy()

# Remove rows with invalid timestamps
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=["Time"]).reset_index(drop=True)

if len(data) < 2:
    raise ValueError("Not enough valid timestamps.")

time = data["Time"].to_numpy(dtype=float)

# Timestamps must be strictly increasing
dt = np.diff(time)

if np.any(dt <= 0):
    raise ValueError(
        "Timestamps are duplicated or not increasing. "
        "Check the original RGB recording."
    )

# ==================================================
# 2. CHECK ORIGINAL FRAME RATE
# ==================================================

original_dt = np.median(dt)
original_fs = 1.0 / original_dt

print("\n========== ORIGINAL RECORDING ==========")
print(f"Original samples: {len(data)}")
print(f"Estimated original FPS: {original_fs:.2f}")
print(f"Median frame interval: {original_dt:.5f} seconds")

# ==================================================
# 3. REMOVE ROWS WITH MISSING RGB VALUES
# ==================================================

rgb_columns = [
    "Forehead_R", "Forehead_G", "Forehead_B",
    "LeftCheek_R", "LeftCheek_G", "LeftCheek_B",
    "RightCheek_R", "RightCheek_G", "RightCheek_B"
]

before_drop = len(data)

data = data.dropna(subset=rgb_columns).reset_index(drop=True)

removed_rows = before_drop - len(data)

print(f"Rows removed due to missing RGB: {removed_rows}")

if len(data) < 2:
    raise ValueError("Not enough valid RGB samples.")

time = data["Time"].to_numpy(dtype=float)

dt = np.diff(time)

if np.any(dt <= 0):
    raise ValueError("Invalid timestamp sequence after cleaning.")

# ==================================================
# 4. IDENTIFY CONTINUOUS SEGMENTS
# ==================================================

MAX_GAP = 0.10

gap_indices = np.where(dt > MAX_GAP)[0]

starts = np.concatenate(([0], gap_indices + 1))
ends = np.concatenate((gap_indices, [len(time) - 1]))

segments = []

for start, end in zip(starts, ends):

    duration = time[end] - time[start]
    samples = end - start + 1

    segments.append({
        "start": start,
        "end": end,
        "duration": duration,
        "samples": samples
    })

if not segments:
    raise ValueError("No continuous recording segment found.")

# ==================================================
# 5. SELECT LONGEST CONTINUOUS SEGMENT
# ==================================================

best_segment = max(
    segments,
    key=lambda segment: segment["duration"]
)

start = best_segment["start"]
end = best_segment["end"]

segment_data = data.iloc[start:end + 1].copy()

segment_time = segment_data["Time"].to_numpy(dtype=float)

duration = segment_time[-1] - segment_time[0]

print("\n========== SELECTED SEGMENT ==========")
print(f"Starting time: {segment_time[0]:.2f} seconds")
print(f"Ending time: {segment_time[-1]:.2f} seconds")
print(f"Duration: {duration:.2f} seconds")
print(f"Original samples: {len(segment_data)}")
print(f"Number of detected gaps: {len(gap_indices)}")

if duration < 10:
    print("WARNING: Segment is shorter than 10 seconds.")
    print("It may not be sufficient for the specified HR analysis.")

# ==================================================
# 6. RESAMPLE TO UNIFORM 30 Hz
# ==================================================

TARGET_FS = 30.0

uniform_time = np.arange(
    segment_time[0],
    segment_time[-1],
    1.0 / TARGET_FS
)

resampled_data = {"Time": uniform_time}

for column in rgb_columns:

    original_values = segment_data[column].to_numpy(dtype=float)

    resampled_values = np.interp(
        uniform_time,
        segment_time,
        original_values
    )

    resampled_data[column] = resampled_values

resampled_df = pd.DataFrame(resampled_data)

if len(resampled_df) < 2:
    raise ValueError("Resampling did not produce enough samples.")

# ==================================================
# 7. SAVE RESAMPLED RGB SIGNALS
# ==================================================

output_file = "rgb_signals_resampled.csv"

resampled_df.to_csv(output_file, index=False)

# ==================================================
# 8. VERIFY RESAMPLING
# ==================================================

new_time = resampled_df["Time"].to_numpy()

new_dt = np.diff(new_time)

actual_fs = 1.0 / np.median(new_dt)

print("\n========== RESAMPLING RESULTS ==========")
print(f"Target sampling frequency: {TARGET_FS:.2f} Hz")
print(f"Actual sampling frequency: {actual_fs:.2f} Hz")
print(f"New number of samples: {len(new_time)}")
print(
    f"New recording duration: "
    f"{new_time[-1] - new_time[0]:.2f} seconds"
)
print(f"Maximum interval: {np.max(new_dt):.5f} seconds")
print(f"Minimum interval: {np.min(new_dt):.5f} seconds")

if not np.allclose(new_dt, 1.0 / TARGET_FS, atol=1e-8):
    raise ValueError("Resampled timestamps are not uniformly spaced.")

print(f"\nResampled RGB signals saved to {output_file}")
print("\nResampling completed successfully!")

