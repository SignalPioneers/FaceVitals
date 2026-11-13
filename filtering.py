
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.signal import butter, sosfiltfilt

# ==========================================
# 1. LOAD POS SIGNALS
# ==========================================

input_file = "pos_signals.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        "pos_signals.csv not found. Run pos_algoritm.py first."
    )

data = pd.read_csv(input_file)

required_columns = [
    "Time",
    "Forehead_POS",
    "Left_Cheek_POS",
    "Right_Cheek_POS"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing required column: {column}")

data = data[required_columns].copy()

# Check timestamps
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=["Time"]).reset_index(drop=True)

time = data["Time"].to_numpy(dtype=float)

if len(time) < 10:
    raise ValueError("Not enough POS samples for filtering.")

# ==========================================
# 2. CHECK SAMPLING FREQUENCY
# ==========================================

time_differences = np.diff(time)

if np.any(time_differences <= 0):
    raise ValueError("Timestamps must be strictly increasing.")

median_dt = np.median(time_differences)
fs = 1.0 / median_dt

print("\n========== SAMPLING INFORMATION ==========")
print(f"Total samples: {len(time)}")
print(f"Median sampling frequency: {fs:.2f} Hz")
print(f"Recording duration: {time[-1] - time[0]:.2f} seconds")

if not np.allclose(
    time_differences,
    median_dt,
    rtol=0,
    atol=1e-6
):
    raise ValueError(
        "Timestamps are not uniformly spaced. "
        "Check the resampled RGB data."
    )

# ==========================================
# 3. DESIGN BUTTERWORTH BANDPASS FILTER
# ==========================================

LOW_CUTOFF = 0.7
HIGH_CUTOFF = 3.5
FILTER_ORDER = 4

nyquist_frequency = fs / 2

if HIGH_CUTOFF >= nyquist_frequency:
    raise ValueError(
        "Upper cutoff frequency must be below Nyquist frequency."
    )

low = LOW_CUTOFF / nyquist_frequency
high = HIGH_CUTOFF / nyquist_frequency

sos = butter(
    FILTER_ORDER,
    [low, high],
    btype="bandpass",
    output="sos"
)

print("\n========== FILTER INFORMATION ==========")
print("Filter type: Butterworth Bandpass")
print(f"Filter order: {FILTER_ORDER}")
print(f"Lower cutoff: {LOW_CUTOFF} Hz")
print(f"Upper cutoff: {HIGH_CUTOFF} Hz")
print(f"Nyquist frequency: {nyquist_frequency:.2f} Hz")

# ==========================================
# 4. FILTER POS SIGNALS
# ==========================================

def apply_bandpass_filter(signal):

    signal = np.asarray(signal, dtype=float)

    # Do not silently remove samples with NaN
    valid_indices = np.isfinite(signal)

    if not np.any(valid_indices):
        raise ValueError("Signal contains no valid samples.")

    # Find the longest continuous valid section
    valid_positions = np.where(valid_indices)[0]

    groups = np.split(
        valid_positions,
        np.where(np.diff(valid_positions) != 1)[0] + 1
    )

    longest_group = max(groups, key=len)

    if len(longest_group) != len(signal):
        print(
            "WARNING: Invalid POS samples detected. "
            "Filtering only the longest continuous valid section."
        )

    clean_signal = signal[longest_group[0]:longest_group[-1] + 1]

    if len(clean_signal) < 10:
        raise ValueError("Not enough valid samples for filtering.")

    filtered_section = sosfiltfilt(sos, clean_signal)

    filtered_signal = np.full(len(signal), np.nan)

    filtered_signal[longest_group[0]:longest_group[-1] + 1] = (
        filtered_section
    )

    return filtered_signal


forehead_filtered = apply_bandpass_filter(
    data["Forehead_POS"]
)

left_cheek_filtered = apply_bandpass_filter(
    data["Left_Cheek_POS"]
)

right_cheek_filtered = apply_bandpass_filter(
    data["Right_Cheek_POS"]
)

print("\nAll three POS signals filtered successfully.")

# ==========================================
# 5. SAVE FILTERED SIGNALS
# ==========================================

filtered_data = pd.DataFrame({
    "Time": time,
    "Forehead_Filtered": forehead_filtered,
    "Left_Cheek_Filtered": left_cheek_filtered,
    "Right_Cheek_Filtered": right_cheek_filtered
})

output_file = "filtered_signals.csv"

filtered_data.to_csv(output_file, index=False)

print(f"\nFiltered signals saved to {output_file}")

# ==========================================
# 6. PLOT FILTERED SIGNALS
# ==========================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 9),
    sharex=True
)

signals = [
    ("Forehead", forehead_filtered, "blue"),
    ("Left Cheek", left_cheek_filtered, "green"),
    ("Right Cheek", right_cheek_filtered, "red")
]

for ax, (name, signal, color) in zip(axes, signals):

    ax.plot(time, signal, color=color)

    ax.set_title(name + " Filtered POS Signal")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()

plt.savefig(
    "filtered_signals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFiltering process completed successfully!")
print("\nGenerated files:")
print("1. filtered_signals.csv")
print("2. filtered_signals.png")
