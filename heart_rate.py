
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================================================
# FACEVITALS - POS HEART RATE ESTIMATION
# Same settings as CHROM for fair comparison
# ==================================================

print("\n==========================================")
print("      FACEVITALS POS HEART RATE")
print("==========================================")

INPUT_FILE = "filtered_signals.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} was not found. Run filtering.py first."
    )

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "Time",
    "Forehead_Filtered",
    "Left_Cheek_Filtered",
    "Right_Cheek_Filtered"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")

if data.empty:
    raise ValueError("Filtered POS file is empty.")

if data[required_columns].isna().any().any():
    raise ValueError(
        "Missing values detected. Do not drop rows blindly."
    )

time = data["Time"].to_numpy(dtype=float)

if not np.all(np.isfinite(time)):
    raise ValueError("Invalid timestamp values.")

dt = np.diff(time)

if len(dt) == 0 or np.any(dt <= 0):
    raise ValueError("Timestamps must be strictly increasing.")

fs = 1.0 / np.median(dt)

if not np.allclose(dt, 1.0 / fs, rtol=0.01, atol=1e-6):
    raise ValueError("Sampling intervals are not uniform.")

duration = time[-1] - time[0]

print("\n========== RECORDING INFORMATION ==========")
print("Total samples:", len(data))
print("Sampling frequency:", round(fs, 2), "Hz")
print("Recording duration:", round(duration, 2), "seconds")

# ==================================================
# 1. TEAM 2 ANALYSIS PARAMETERS
# ==================================================

MIN_FREQ = 0.7
MAX_FREQ = 3.5

WINDOW_SECONDS = 10
UPDATE_SECONDS = 1

FFT_SIZE = 2048
MEDIAN_HISTORY = 5

CONFIDENCE_THRESHOLD = 0.30
MAX_HR_JUMP = 20.0

window_size = int(round(WINDOW_SECONDS * fs))
update_step = int(round(UPDATE_SECONDS * fs))

if len(data) < window_size:
    raise ValueError(
        "Recording must contain at least 10 seconds of data."
    )

print("\n========== ANALYSIS PARAMETERS ==========")
print("HR range:", MIN_FREQ * 60, "-", MAX_FREQ * 60, "BPM")
print("Window duration:", WINDOW_SECONDS, "seconds")
print("Window samples:", window_size)
print("Update interval:", UPDATE_SECONDS, "second")
print("FFT size:", FFT_SIZE)
print("Median history:", MEDIAN_HISTORY, "readings")

# ==================================================
# 2. HEART RATE AND CONFIDENCE FUNCTION
# ==================================================

def estimate_hr(signal, fs):

    signal = np.asarray(signal, dtype=float)

    if len(signal) != window_size:
        raise ValueError("Incorrect analysis window size.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains invalid values.")

    # Remove DC component
    signal = signal - np.mean(signal)

    # Apply Hann window
    hann_window = np.hanning(len(signal))
    windowed_signal = signal * hann_window

    # Calculate FFT with zero-padding
    spectrum = np.fft.rfft(
        windowed_signal,
        n=FFT_SIZE
    )

    frequencies = np.fft.rfftfreq(
        FFT_SIZE,
        d=1.0 / fs
    )

    power = np.abs(spectrum) ** 2

    # Select HR frequency band
    band_mask = (
        (frequencies >= MIN_FREQ) &
        (frequencies <= MAX_FREQ)
    )

    if not np.any(band_mask):
        raise ValueError("No frequencies in HR range.")

    band_frequencies = frequencies[band_mask]
    band_power = power[band_mask]

    # Find dominant frequency
    peak_index = np.argmax(band_power)

    dominant_frequency = band_frequencies[peak_index]

    raw_hr = dominant_frequency * 60.0

    # Fundamental frequency power
    fundamental_mask = (
        np.abs(frequencies - dominant_frequency) <= 0.1
    )

    fundamental_power = np.sum(
        power[fundamental_mask]
    )

    # Second harmonic power
    second_harmonic = 2.0 * dominant_frequency

    harmonic_mask = (
        (frequencies >= MIN_FREQ) &
        (frequencies <= MAX_FREQ) &
        (np.abs(frequencies - second_harmonic) <= 0.1)
    )

    harmonic_power = np.sum(
        power[harmonic_mask]
    )

    total_band_power = np.sum(band_power)

    if total_band_power <= 0:
        confidence = 0.0
    else:
        confidence = (
            fundamental_power + harmonic_power
        ) / total_band_power

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    return (
        raw_hr,
        dominant_frequency,
        confidence,
        frequencies,
        power
    )


# ==================================================
# 3. PROCESS THREE FACIAL REGIONS
# ==================================================

regions = {
    "Forehead": "Forehead_Filtered",
    "Left Cheek": "Left_Cheek_Filtered",
    "Right Cheek": "Right_Cheek_Filtered"
}

hr_history = {
    region: [] for region in regions
}

previous_hr = {
    region: None for region in regions
}

results = []
latest_spectra = {}

# Start after 10-second warm-up
for end_index in range(
    window_size,
    len(data) + 1,
    update_step
):

    start_index = end_index - window_size
    window_time = time[end_index - 1]

    row = {"Time": window_time}
    valid_region_hrs = []

    for region_name, column in regions.items():

        signal = data[column].to_numpy(dtype=float)[
            start_index:end_index
        ]

        (
            raw_hr,
            dominant_frequency,
            confidence,
            frequencies,
            power
        ) = estimate_hr(signal, fs)

        # Median of the latest five readings
        hr_history[region_name].append(raw_hr)

        if len(hr_history[region_name]) > MEDIAN_HISTORY:
            hr_history[region_name].pop(0)

        smoothed_hr = float(
            np.median(hr_history[region_name])
        )

        # Detect sudden HR changes
        jump_warning = False

        if previous_hr[region_name] is not None:

            hr_change = abs(
                smoothed_hr - previous_hr[region_name]
            )

            if hr_change > MAX_HR_JUMP:
                jump_warning = True

        previous_hr[region_name] = smoothed_hr

        # Determine status
        if confidence < CONFIDENCE_THRESHOLD:
            status = "low_confidence"

        elif jump_warning:
            status = "jump_warning"

        else:
            status = "ok"

        if status == "ok":
            valid_region_hrs.append(smoothed_hr)

        row[f"{region_name}_Raw_HR"] = raw_hr
        row[f"{region_name}_Smoothed_HR"] = smoothed_hr
        row[f"{region_name}_Confidence"] = confidence
        row[f"{region_name}_Status"] = status

        latest_spectra[region_name] = {
            "HR": raw_hr,
            "Frequency": frequencies,
            "Power": power,
            "Dominant": dominant_frequency
        }

    # Combined HR from reliable regions only
    if len(valid_region_hrs) > 0:

        combined_hr = float(
            np.median(valid_region_hrs)
        )

        combined_status = "ok"

    else:

        combined_hr = np.nan
        combined_status = "low_confidence"

    row["Combined_HR"] = combined_hr
    row["Combined_Status"] = combined_status

    results.append(row)


# ==================================================
# 4. DISPLAY LATEST RESULTS
# ==================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No HR estimates were generated.")

latest = results_df.iloc[-1]

print("\n==========================================")
print("       POS HEART RATE RESULTS")
print("==========================================")

for region_name in regions:

    print(f"\n{region_name}:")

    print(
        "Raw HR:",
        round(latest[f"{region_name}_Raw_HR"], 2),
        "BPM"
    )

    print(
        "Smoothed HR:",
        round(latest[f"{region_name}_Smoothed_HR"], 2),
        "BPM"
    )

    print(
        "Confidence:",
        round(latest[f"{region_name}_Confidence"], 3)
    )

    print(
        "Status:",
        latest[f"{region_name}_Status"]
    )

print("\n------------------------------------------")

if np.isfinite(latest["Combined_HR"]):

    print(
        "Combined POS HR:",
        round(latest["Combined_HR"], 2),
        "BPM"
    )

else:

    print("Combined POS HR: Not reliable")

print("Combined status:", latest["Combined_Status"])

print("\n==========================================")


# ==================================================
# 5. SAVE ALL HR ESTIMATES
# ==================================================

OUTPUT_FILE = "heart_rate_results.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nAll rolling HR estimates saved to:", OUTPUT_FILE)


# ==================================================
# 6. PLOT LATEST FREQUENCY SPECTRA
# ==================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10)
)

for ax, (region_name, result) in zip(
    axes,
    latest_spectra.items()
):

    frequencies = result["Frequency"]
    power = result["Power"]

    valid = (
        (frequencies >= MIN_FREQ) &
        (frequencies <= MAX_FREQ)
    )

    ax.plot(
        frequencies[valid],
        power[valid],
        linewidth=1.5
    )

    ax.axvline(
        result["Dominant"],
        linestyle="--",
        label=f"HR = {result['HR']:.2f} BPM"
    )

    ax.set_title(
        region_name + " POS Frequency Spectrum"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True)

plt.tight_layout()

plt.savefig(
    "heart_rate_spectrum.png",
    dpi=200,
    bbox_inches="tight"
)

print("Latest POS spectrum graph saved successfully.")

plt.show(block=True)


# ==================================================
# 7. FINAL SUMMARY
# ==================================================

print("\n==========================================")
print("        POS HEART RATE SUMMARY")
print("==========================================")

print("Analysis window:", WINDOW_SECONDS, "seconds")
print("Update interval:", UPDATE_SECONDS, "second")
print("FFT size:", FFT_SIZE)
print("Median smoothing:", MEDIAN_HISTORY, "readings")
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("Number of HR estimates:", len(results_df))

print("\nGenerated files:")
print("1. heart_rate_results.csv")
print("2. heart_rate_spectrum.png")

print("\nNOTE:")
print("Confidence is a spectral quality score,")
print("not a probability that the HR is correct.")
print("Clinical validation requires a reference device.")

print("\nFACEVITALS POS HEART RATE COMPLETED!")

