
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.signal import butter, sosfiltfilt

# ==================================================
# 1. LOAD RESAMPLED RGB DATA
# ==================================================

print("\n========== FACEVITALS CHROM PROCESSING ==========\n")

INPUT_FILE = "rgb_signals_resampled.csv"
OUTPUT_FILE = "chrom_signals.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} was not found!")

data = pd.read_csv(INPUT_FILE)

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

data = data[required_columns].copy()

if not np.all(np.isfinite(data.to_numpy(dtype=float))):
    raise ValueError("RGB data contains missing or invalid values.")

time = data["Time"].to_numpy(dtype=float)

if len(time) < 48:
    raise ValueError("Not enough RGB samples for CHROM processing.")

time_differences = np.diff(time)

if np.any(time_differences <= 0):
    raise ValueError("Timestamps must be strictly increasing.")

median_dt = np.median(time_differences)
fs = 1.0 / median_dt

if not np.allclose(
    time_differences,
    median_dt,
    rtol=0,
    atol=1e-6
):
    raise ValueError("RGB timestamps are not uniformly spaced.")

print("Total measurements:", len(data))
print("Sampling frequency:", round(fs, 2), "Hz")
print(
    "Recording duration:",
    round(time[-1] - time[0], 2),
    "seconds"
)

# ==================================================
# 2. DESIGN BUTTERWORTH BANDPASS FILTER
# ==================================================

LOW_CUTOFF = 0.7
HIGH_CUTOFF = 3.5
FILTER_ORDER = 4

nyquist_frequency = fs / 2

if HIGH_CUTOFF >= nyquist_frequency:
    raise ValueError("Upper cutoff must be below Nyquist frequency.")

low = LOW_CUTOFF / nyquist_frequency
high = HIGH_CUTOFF / nyquist_frequency

sos = butter(
    FILTER_ORDER,
    [low, high],
    btype="bandpass",
    output="sos"
)

print("\n========== CHROM FILTER INFORMATION ==========")
print("Filter: Butterworth Bandpass")
print("Filter order:", FILTER_ORDER)
print("Lower cutoff:", LOW_CUTOFF, "Hz")
print("Upper cutoff:", HIGH_CUTOFF, "Hz")

# ==================================================
# 3. CHROM ALGORITHM FUNCTION
# ==================================================

def apply_chrom(
    red,
    green,
    blue,
    fs,
    window_seconds=1.6
):

    red = np.asarray(red, dtype=float)
    green = np.asarray(green, dtype=float)
    blue = np.asarray(blue, dtype=float)

    total_samples = len(red)

    # Team 2: 1.6-second window
    window_size = int(round(window_seconds * fs))

    if total_samples < window_size:
        raise ValueError("Recording is shorter than the CHROM window.")

    # Team 2: stride = 1 sample
    step = 1

    chrom_signal = np.zeros(total_samples)
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

        # A. NORMALIZE RGB CHANNELS

        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        if min(mean_R, mean_G, mean_B) <= 1e-10:
            continue

        Rn = R / mean_R
        Gn = G / mean_G
        Bn = B / mean_B

        # B. CALCULATE CHROMINANCE SIGNALS

        X = 3.0 * Rn - 2.0 * Gn

        Y = 1.5 * Rn + Gn - 1.5 * Bn

        # C. FILTER X AND Y BEFORE ALPHA

        X_filtered = sosfiltfilt(sos, X)
        Y_filtered = sosfiltfilt(sos, Y)

        # D. CALCULATE ALPHA AFTER FILTERING

        std_X = np.std(X_filtered)
        std_Y = np.std(Y_filtered)

        if std_Y < 1e-10:
            continue

        alpha = std_X / std_Y

        # E. CALCULATE CHROM PULSE SIGNAL

        h = X_filtered - alpha * Y_filtered

        # Remove DC component
        h = h - np.mean(h)

        # F. OVERLAP-ADD

        chrom_signal[start:end] += h
        overlap_count[start:end] += 1

    # Average overlapping windows
    valid = overlap_count > 0

    chrom_signal[valid] /= overlap_count[valid]

    return chrom_signal, valid


# ==================================================
# 4. PROCESS THREE FACIAL REGIONS
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

    signal, valid = apply_chrom(
        red,
        green,
        blue,
        fs
    )

    results[region_name] = (signal, valid)

    print(region_name, "CHROM processing completed.")


# ==================================================
# 5. SAVE CHROM SIGNALS
# ==================================================

output = pd.DataFrame({
    "Time": time
})

for region_name, (signal, valid) in results.items():

    column_name = region_name.replace(" ", "_") + "_CHROM"

    output[column_name] = np.where(
        valid,
        signal,
        np.nan
    )

output.to_csv(OUTPUT_FILE, index=False)

print("\nCHROM measurements saved in", OUTPUT_FILE)


# ==================================================
# 6. PLOT CHROM SIGNALS
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
        linewidth=1
    )

    ax.set_title(region_name + " CHROM Pulse Signal")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "FaceVitals - CHROM Signal Extraction",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "chrom_signals_complete.png",
    dpi=200,
    bbox_inches="tight"
)

print("CHROM graphs saved successfully.")

plt.show(block=True)


# ==================================================
# 7. FINAL SUMMARY
# ==================================================

print("\n========== CHROM PROCESSING SUMMARY ==========")

print("Sampling frequency:", round(fs, 2), "Hz")
print("Window duration: 1.6 seconds")
print("Window size:", int(round(1.6 * fs)), "samples")
print("Stride: 1 sample")
print("Filter order:", FILTER_ORDER)
print("Frequency range:", LOW_CUTOFF, "-", HIGH_CUTOFF, "Hz")

print("Forehead CHROM: Completed")
print("Left Cheek CHROM: Completed")
print("Right Cheek CHROM: Completed")

print("\nGenerated files:")
print("1. chrom_signals.csv")
print("2. chrom_signals_complete.png")

print("\nFaceVitals CHROM processing finished!")

