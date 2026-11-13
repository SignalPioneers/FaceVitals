
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================================================
# FACEVITALS - CHROM SIGNAL VALIDATION
# X AND Y FILTERING IS ALREADY DONE IN chrom_algorithm.py
# ==================================================

print("\n==========================================")
print("       FACEVITALS CHROM VALIDATION")
print("==========================================")

# ==================================================
# 1. LOAD CHROM SIGNALS
# ==================================================

INPUT_FILE = "chrom_signals.csv"
OUTPUT_FILE = "chrom_filtered_signals.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"ERROR: {INPUT_FILE} was not found!"
    )

data = pd.read_csv(INPUT_FILE)

if data.empty:
    raise ValueError("ERROR: CHROM file is empty!")

required_columns = [
    "Time",
    "Forehead_CHROM",
    "Left_Cheek_CHROM",
    "Right_Cheek_CHROM"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(
            f"ERROR: Missing column: {column}"
        )

# Do not drop rows because this could disturb uniform sampling
if data[required_columns].isna().any().any():
    raise ValueError(
        "ERROR: Missing values detected in CHROM signals!"
    )

time = data["Time"].to_numpy(dtype=float)

# ==================================================
# 2. CHECK SAMPLING INFORMATION
# ==================================================

time_differences = np.diff(time)

if np.any(time_differences <= 0):
    raise ValueError("ERROR: Invalid timestamps!")

fs = 1.0 / np.median(time_differences)

if not np.allclose(
    time_differences,
    1.0 / fs,
    rtol=0.01,
    atol=1e-6
):
    raise ValueError(
        "ERROR: Sampling intervals are not uniform!"
    )

duration = time[-1] - time[0]

print("\n========== SAMPLING INFORMATION ==========")
print("Total samples:", len(data))
print("Sampling frequency:", round(fs, 2), "Hz")
print("Recording duration:", round(duration, 2), "seconds")

# ==================================================
# 3. VALIDATE CHROM SIGNALS
# ==================================================

regions = {
    "Forehead": "Forehead_CHROM",
    "Left Cheek": "Left_Cheek_CHROM",
    "Right Cheek": "Right_Cheek_CHROM"
}

validated_results = {}

for region_name, column in regions.items():

    signal = data[column].to_numpy(dtype=float)

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"ERROR: Invalid values in {column}"
        )

    if np.std(signal) < 1e-12:
        raise ValueError(
            f"ERROR: {region_name} CHROM signal is nearly constant!"
        )

    validated_results[region_name] = signal

    print(region_name, "CHROM signal validated.")

# ==================================================
# 4. SAVE SIGNALS
# ==================================================

# X and Y were already bandpass-filtered in chrom_algorithm.py.
# Therefore, no second filtering is applied here.

output = pd.DataFrame({
    "Time": time,
    "Forehead_CHROM_Filtered":
        validated_results["Forehead"],
    "Left_Cheek_CHROM_Filtered":
        validated_results["Left Cheek"],
    "Right_Cheek_CHROM_Filtered":
        validated_results["Right Cheek"]
})

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCHROM signals saved to:", OUTPUT_FILE)

# ==================================================
# 5. PLOT CHROM SIGNALS
# ==================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 10),
    sharex=True
)

for ax, (region_name, signal) in zip(
    axes,
    validated_results.items()
):

    ax.plot(
        time,
        signal,
        linewidth=1
    )

    ax.set_title(
        region_name + " CHROM Signal (X/Y Filtered Upstream)"
    )

    ax.set_ylabel("Amplitude")
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "FaceVitals - Validated CHROM Signals",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "chrom_filtered_signals.png",
    dpi=200,
    bbox_inches="tight"
)

print("CHROM signal graph saved successfully.")

plt.show(block=True)

# ==================================================
# 6. FINAL SUMMARY
# ==================================================

print("\n==========================================")
print("       CHROM VALIDATION SUMMARY")
print("==========================================")

print("Forehead: Validated")
print("Left Cheek: Validated")
print("Right Cheek: Validated")

print("\nImportant:")
print("No second bandpass filtering was applied.")
print("X and Y filtering was already performed")
print("inside chrom_algorithm.py.")

print("\nGenerated files:")
print("1. chrom_filtered_signals.csv")
print("2. chrom_filtered_signals.png")

print("\nFACEVITALS CHROM VALIDATION COMPLETED!")

