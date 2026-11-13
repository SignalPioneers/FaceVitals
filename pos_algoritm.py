
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================================================
# 1. LOAD RESAMPLED RGB DATA
# ==================================================

print("\n========== FACEVITALS POS PROCESSING ==========\n")

input_file = "rgb_signals_resampled.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"{input_file} was not found!")

data = pd.read_csv(input_file)

if data.empty:
    raise ValueError("The RGB file contains no measurements.")

required_columns = [
    "Time",
    "Forehead_R", "Forehead_G", "Forehead_B",
    "LeftCheek_R", "LeftCheek_G", "LeftCheek_B",
    "RightCheek_R", "RightCheek_G", "RightCheek_B"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")

data = data.dropna(subset=required_columns).reset_index(drop=True)

if len(data) < 48:
    raise ValueError("At least 48 valid samples are required.")

time = data["Time"].to_numpy(dtype=float)

time_differences = np.diff(time)

if np.any(time_differences <= 0):
    raise ValueError("Timestamps must be strictly increasing.")

fs = 1.0 / np.median(time_differences)

print("Total measurements:", len(data))
print("Estimated sampling frequency:", round(fs, 2), "Hz")
print(
    "Recording duration:",
    round(time[-1] - time[0], 2),
    "seconds"
)

# ==================================================
# 2. POS ALGORITHM FUNCTION
# ==================================================

def apply_pos(red, green, blue, fs, window_seconds=1.6):

    red = np.asarray(red, dtype=float)
    green = np.asarray(green, dtype=float)
    blue = np.asarray(blue, dtype=float)

    total_samples = len(red)

    # Team 2 specification:
    # 1.6 seconds x 30 Hz = 48 samples
    window_size = int(round(window_seconds * fs))

    if total_samples < window_size:
        raise ValueError("Not enough samples for POS window.")

    # Team 2 specification: stride = 1 sample
    step = 1

    # Output arrays for overlap-add
    pulse_signal = np.zeros(total_samples)
    overlap_count = np.zeros(total_samples)

    # Process overlapping windows
    for start in range(
        0,
        total_samples - window_size + 1,
        step
    ):

        end = start + window_size

        R = red[start:end]
        G = green[start:end]
        B = blue[start:end]

        # A. NORMALIZE RGB SIGNALS

        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        if min(mean_R, mean_G, mean_B) <= 1e-10:
            continue

        Rn = R / mean_R
        Gn = G / mean_G
        Bn = B / mean_B

        # B. CALCULATE PROJECTED SIGNALS

        S1 = Gn - Bn
        S2 = Gn + Bn - 2 * Rn

        # C. CALCULATE ALPHA

        std_S1 = np.std(S1)
        std_S2 = np.std(S2)

        if std_S2 < 1e-10:
            continue

        alpha = std_S1 / std_S2

        # D. CALCULATE POS SIGNAL

        h = S1 + alpha * S2

        # Remove DC component
        h = h - np.mean(h)

        # E. OVERLAP-ADD

        pulse_signal[start:end] += h
        overlap_count[start:end] += 1

    # Average overlapping windows
    valid = overlap_count > 0

    pulse_signal[valid] /= overlap_count[valid]

    return pulse_signal, valid


# ==================================================
# 3. PROCESS THE THREE FACIAL REGIONS
# ==================================================

regions = {
    "Forehead": (
        "Forehead_R",
        "Forehead_G",
        "Forehead_B"
    ),

    "Left Cheek": (
        "LeftCheek_R",
        "LeftCheek_G",
        "LeftCheek_B"
    ),

    "Right Cheek": (
        "RightCheek_R",
        "RightCheek_G",
        "RightCheek_B"
    )
}

results = {}

for region_name, columns in regions.items():

    red = data[columns[0]].to_numpy(dtype=float)
    green = data[columns[1]].to_numpy(dtype=float)
    blue = data[columns[2]].to_numpy(dtype=float)

    signal, valid = apply_pos(
        red,
        green,
        blue,
        fs
    )

    results[region_name] = (signal, valid)

    print(region_name, "POS processing completed.")


# ==================================================
# 4. SAVE POS SIGNALS TO CSV
# ==================================================

output = pd.DataFrame({
    "Time": time
})

for region_name, (signal, valid) in results.items():

    column_name = region_name.replace(" ", "_") + "_POS"

    output[column_name] = np.where(
        valid,
        signal,
        np.nan
    )

output.to_csv("pos_signals.csv", index=False)

print("\nPOS measurements saved in pos_signals.csv")


# ==================================================
# 5. PLOT ALL THREE POS SIGNALS
# ==================================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(13, 10),
    sharex=True
)

for ax, (region_name, result) in zip(
    axes, results.items()
):

    signal, valid = result

    ax.plot(
        time[valid],
        signal[valid],
        color="purple",
        linewidth=1
    )

    ax.set_title(region_name + " POS Pulse Signal")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "FaceVitals - POS Signal Extraction",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "pos_signals_complete.png",
    dpi=200,
    bbox_inches="tight"
)

print("Complete POS graphs saved successfully.")

plt.show(block=True)


# ==================================================
# 6. DISPLAY FINAL INFORMATION
# ==================================================

print("\n========== PROCESSING SUMMARY ==========")

print("Sampling frequency:", round(fs, 2), "Hz")
print("Window duration: 1.6 seconds")
print("Window size:", int(round(1.6 * fs)), "samples")
print("Stride: 1 sample")

print("Forehead POS: Completed")
print("Left Cheek POS: Completed")
print("Right Cheek POS: Completed")

print("\nGenerated files:")
print("1. pos_signals.csv")
print("2. pos_signals_complete.png")

print("\nFaceVitals POS processing finished!")

