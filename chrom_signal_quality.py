
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import periodogram, find_peaks

# ============================================================
# FACEVITALS: CHROM SIGNAL QUALITY ASSESSMENT
# ============================================================

INPUT_FILE = "chrom_filtered_signals.csv"
OUTPUT_FILE = "chrom_signal_quality_results.csv"
PLOT_FILE = "chrom_signal_quality_scores.png"

MIN_HR_FREQ = 0.7
MAX_HR_FREQ = 3.0

WINDOW_SECONDS = 10
WINDOW_STEP_SECONDS = 5

# ============================================================
# 1. LOAD CHROM FILTERED SIGNALS
# ============================================================

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "Time",
    "Forehead_CHROM_Filtered",
    "Left_Cheek_CHROM_Filtered",
    "Right_Cheek_CHROM_Filtered"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")

data = data[required_columns].apply(
    pd.to_numeric, errors="coerce"
)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna().reset_index(drop=True)

if len(data) < 250:
    raise ValueError("Not enough valid samples.")

time = data["Time"].to_numpy()
dt = np.diff(time)

if np.any(dt <= 0):
    raise ValueError("Invalid timestamps.")

median_dt = np.median(dt)
fs = 1.0 / median_dt

if np.max(np.abs(dt - median_dt)) > 0.05 * median_dt:
    raise ValueError(
        "Sampling is not uniform. Check resampling."
    )

print("\n==========================================")
print("       CHROM SIGNAL QUALITY")
print("==========================================")
print(f"Total samples: {len(data)}")
print(f"Sampling frequency: {fs:.2f} Hz")
print(f"Duration: {time[-1] - time[0]:.2f} seconds")


# ============================================================
# 2. QUALITY SCORE FUNCTION
# ============================================================

def calculate_score(value, minimum, maximum):

    if not np.isfinite(value):
        return 0.0

    score = (
        (value - minimum) /
        (maximum - minimum)
    ) * 100

    return float(np.clip(score, 0, 100))


# ============================================================
# 3. AMPLITUDE STABILITY
# ============================================================

def amplitude_stability(signal):

    window_size = int(5 * fs)
    rms_values = []

    for start in range(
        0, len(signal) - window_size + 1, window_size
    ):

        segment = signal[start:start + window_size]

        rms = np.sqrt(np.mean(segment ** 2))
        rms_values.append(rms)

    if len(rms_values) < 2:
        return np.nan, 0.0

    mean_rms = np.mean(rms_values)

    if mean_rms <= 1e-12:
        return np.inf, 0.0

    cv = np.std(rms_values) / mean_rms

    score = 100 * np.exp(-cv)

    return float(cv), float(score)


# ============================================================
# 4. HEART RATE STABILITY
# ============================================================

def heart_rate_stability(signal):

    window_size = int(WINDOW_SECONDS * fs)
    step_size = int(WINDOW_STEP_SECONDS * fs)

    estimates = []

    for start in range(
        0, len(signal) - window_size + 1, step_size
    ):

        segment = signal[start:start + window_size]

        frequencies, power = periodogram(
            segment,
            fs=fs,
            window="hann",
            detrend="constant",
            scaling="density"
        )

        mask = (
            (frequencies >= MIN_HR_FREQ) &
            (frequencies <= MAX_HR_FREQ)
        )

        if not np.any(mask):
            continue

        valid_frequencies = frequencies[mask]
        valid_power = power[mask]

        peak_index = np.argmax(valid_power)

        estimated_hr = valid_frequencies[peak_index] * 60

        estimates.append(estimated_hr)

    if len(estimates) < 2:
        return estimates, np.nan, 0.0

    hr_std = np.std(estimates)

    score = max(0.0, 100.0 - 8.0 * hr_std)

    return estimates, float(hr_std), float(score)


# ============================================================
# 5. ANALYSE ONE CHROM FACIAL REGION
# ============================================================

def analyse_region(signal):

    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    frequencies, power = periodogram(
        signal,
        fs=fs,
        window="hann",
        detrend="constant",
        scaling="density"
    )

    band_mask = (
        (frequencies >= MIN_HR_FREQ) &
        (frequencies <= MAX_HR_FREQ)
    )

    band_frequencies = frequencies[band_mask]
    band_power = power[band_mask]

    if len(band_power) < 3:
        raise ValueError("Insufficient frequency resolution.")

    # --------------------------------------------------------
    # DOMINANT FREQUENCY AND HEART RATE
    # --------------------------------------------------------

    peak_index = np.argmax(band_power)

    dominant_frequency = band_frequencies[peak_index]
    heart_rate = dominant_frequency * 60
    peak_power = band_power[peak_index]

    # --------------------------------------------------------
    # PARAMETER 1: SIGNAL-TO-NOISE RATIO
    # --------------------------------------------------------

    noise_floor = np.median(band_power)

    if noise_floor > 1e-15:

        snr_db = 10 * np.log10(
            max(peak_power, 1e-15) / noise_floor
        )

    else:
        snr_db = np.nan

    snr_score = calculate_score(snr_db, -5, 15)

    # --------------------------------------------------------
    # PARAMETER 2: SPECTRAL CONCENTRATION
    # --------------------------------------------------------

    peak_region = (
        np.abs(band_frequencies - dominant_frequency) <= 0.1
    )

    total_power = np.sum(band_power)

    concentrated_power = np.sum(
        band_power[peak_region]
    )

    if total_power > 1e-15:
        concentration = concentrated_power / total_power
    else:
        concentration = 0.0

    concentration_score = concentration * 100

    # --------------------------------------------------------
    # PARAMETER 3: PEAK PROMINENCE
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        band_power,
        prominence=0
    )

    if len(peaks) > 0:

        closest_peak = np.argmin(
            np.abs(peaks - peak_index)
        )

        prominence = properties[
            "prominences"
        ][closest_peak]

        prominence_ratio = (
            prominence / max(peak_power, 1e-15)
        )

    else:

        prominence = 0.0
        prominence_ratio = 0.0

    prominence_score = calculate_score(
        prominence_ratio, 0, 0.5
    )

    # --------------------------------------------------------
    # PARAMETER 4: AMPLITUDE STABILITY
    # --------------------------------------------------------

    amplitude_cv, amplitude_score = amplitude_stability(
        signal
    )

    # --------------------------------------------------------
    # PARAMETER 5: HEART RATE STABILITY
    # --------------------------------------------------------

    window_hr, hr_std, hr_stability_score = (
        heart_rate_stability(signal)
    )

    return {

        "HR_BPM": heart_rate,
        "Dominant_Frequency_Hz": dominant_frequency,

        "SNR_dB": snr_db,
        "SNR_Score": snr_score,

        "Spectral_Concentration": concentration,
        "Spectral_Score": concentration_score,

        "Peak_Prominence": prominence,
        "Peak_Prominence_Score": prominence_score,

        "Amplitude_CV": amplitude_cv,
        "Amplitude_Stability_Score": amplitude_score,

        "Window_HR_Values": window_hr,
        "HR_Standard_Deviation": hr_std,
        "HR_Stability_Score": hr_stability_score
    }


# ============================================================
# 6. PROCESS ALL FACIAL REGIONS
# ============================================================

regions = {

    "Forehead": data[
        "Forehead_CHROM_Filtered"
    ].to_numpy(),

    "Left Cheek": data[
        "Left_Cheek_CHROM_Filtered"
    ].to_numpy(),

    "Right Cheek": data[
        "Right_Cheek_CHROM_Filtered"
    ].to_numpy()
}

results = {}

for region_name, signal in regions.items():

    results[region_name] = analyse_region(signal)

    print(f"{region_name} analysis completed.")


# ============================================================
# 7. PARAMETER 6: INTER-REGION AGREEMENT
# ============================================================

hr_values = np.array([
    results[name]["HR_BPM"]
    for name in regions
])

median_hr = np.median(hr_values)

for region_name in regions:

    region_hr = results[region_name]["HR_BPM"]

    difference = abs(region_hr - median_hr)

    agreement_score = max(
        0.0,
        100.0 - difference * 5.0
    )

    results[region_name][
        "HR_Difference_From_Median"
    ] = difference

    results[region_name][
        "Inter_Region_Agreement_Score"
    ] = agreement_score


# ============================================================
# 8. OVERALL QUALITY SCORE
# ============================================================

weights = {

    "SNR_Score": 0.20,
    "Spectral_Score": 0.20,
    "Peak_Prominence_Score": 0.15,
    "Amplitude_Stability_Score": 0.15,
    "HR_Stability_Score": 0.15,
    "Inter_Region_Agreement_Score": 0.15
}

for region_name in regions:

    result = results[region_name]

    quality_score = sum(
        result[key] * weight
        for key, weight in weights.items()
    )

    result["Overall_Quality_Score"] = quality_score

    if quality_score >= 75:
        status = "GOOD"

    elif quality_score >= 50:
        status = "MODERATE"

    else:
        status = "POOR"

    result["Quality_Status"] = status


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("       CHROM SIGNAL QUALITY RESULTS")
print("==========================================")

for region_name in regions:

    result = results[region_name]

    print(f"\n--- {region_name.upper()} ---")

    print(f"Estimated HR: {result['HR_BPM']:.2f} BPM")

    print(
        f"Dominant Frequency: "
        f"{result['Dominant_Frequency_Hz']:.3f} Hz"
    )

    print(f"SNR: {result['SNR_dB']:.2f} dB")
    print(f"SNR Score: {result['SNR_Score']:.2f}/100")

    print(
        f"Spectral Concentration: "
        f"{result['Spectral_Concentration']:.3f}"
    )

    print(
        f"Spectral Score: "
        f"{result['Spectral_Score']:.2f}/100"
    )

    print(
        f"Peak Prominence: "
        f"{result['Peak_Prominence']:.6f}"
    )

    print(
        f"Peak Prominence Score: "
        f"{result['Peak_Prominence_Score']:.2f}/100"
    )

    print(f"Amplitude CV: {result['Amplitude_CV']:.4f}")

    print(
        f"Amplitude Stability Score: "
        f"{result['Amplitude_Stability_Score']:.2f}/100"
    )

    print(
        f"HR Standard Deviation: "
        f"{result['HR_Standard_Deviation']:.2f} BPM"
    )

    print(
        f"HR Stability Score: "
        f"{result['HR_Stability_Score']:.2f}/100"
    )

    print(
        f"Inter-Region Agreement Score: "
        f"{result['Inter_Region_Agreement_Score']:.2f}/100"
    )

    print(
        f"OVERALL QUALITY: "
        f"{result['Overall_Quality_Score']:.2f}/100"
    )

    print(f"STATUS: {result['Quality_Status']}")


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_rows = []

for region_name, result in results.items():

    row = {"Region": region_name}

    for key, value in result.items():

        if key == "Window_HR_Values":

            row[key] = "; ".join(
                f"{hr:.2f}" for hr in value
            )

        else:

            row[key] = value

    output_rows.append(row)

results_df = pd.DataFrame(output_rows)

results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nResults saved to {OUTPUT_FILE}")


# ============================================================
# 11. PLOT QUALITY SCORES
# ============================================================

region_names = list(results.keys())

quality_scores = [
    results[name]["Overall_Quality_Score"]
    for name in region_names
]

bar_colors = []

for score in quality_scores:

    if score >= 75:
        bar_colors.append("green")

    elif score >= 50:
        bar_colors.append("orange")

    else:
        bar_colors.append("red")


plt.figure(figsize=(10, 6))

bars = plt.bar(
    region_names,
    quality_scores,
    color=bar_colors,
    edgecolor="black"
)

plt.ylim(0, 110)

plt.ylabel("Quality Score (0-100)")
plt.xlabel("Facial Region")

plt.title(
    "FaceVitals - CHROM Facial Signal Quality Assessment"
)

plt.axhline(
    y=75,
    color="green",
    linestyle="--",
    label="Good Threshold (75)"
)

plt.axhline(
    y=50,
    color="orange",
    linestyle="--",
    label="Moderate Threshold (50)"
)

for bar, score in zip(bars, quality_scores):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 2,
        f"{score:.1f}",
        ha="center",
        fontweight="bold"
    )

plt.legend()
plt.tight_layout()

plt.savefig(PLOT_FILE, dpi=300)
plt.show()

print("\n==========================================")
print("CHROM signal quality assessment completed!")
print("==========================================")
