
import pandas as pd
import numpy as np
import os

# ==========================================
# FACEVITALS - CHROM STABILITY ANALYSIS
# ==========================================

INPUT_FILE = "chrom_heart_rate_results.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} not found!")

data = pd.read_csv(INPUT_FILE)

regions = {
    "Forehead": "Forehead",
    "Left Cheek": "Left Cheek",
    "Right Cheek": "Right Cheek"
}

results = []

print("\n==========================================")
print("       CHROM STABILITY ANALYSIS")
print("==========================================")

for region_name, prefix in regions.items():

    hr_column = f"{prefix}_Smoothed_HR"
    confidence_column = f"{prefix}_Confidence"
    status_column = f"{prefix}_Status"

    hr = pd.to_numeric(
        data[hr_column], errors="coerce"
    ).dropna()

    confidence = pd.to_numeric(
        data[confidence_column], errors="coerce"
    ).dropna()

    status = data[status_column].astype(str)

    if len(hr) == 0:
        print(f"\n{region_name}: No valid HR readings.")
        continue

    mean_hr = hr.mean()
    std_hr = hr.std()
    min_hr = hr.min()
    max_hr = hr.max()

    mean_confidence = confidence.mean()

    ok_percentage = (
        (status == "ok").sum() / len(status)
    ) * 100

    results.append({
        "Region": region_name,
        "Mean_HR_BPM": mean_hr,
        "Std_HR_BPM": std_hr,
        "Minimum_HR_BPM": min_hr,
        "Maximum_HR_BPM": max_hr,
        "Mean_Confidence": mean_confidence,
        "OK_Percentage": ok_percentage,
        "Number_of_Readings": len(hr)
    })

    print(f"\n--- {region_name} ---")
    print(f"Mean HR: {mean_hr:.2f} BPM")
    print(f"Standard deviation: {std_hr:.2f} BPM")
    print(f"Minimum HR: {min_hr:.2f} BPM")
    print(f"Maximum HR: {max_hr:.2f} BPM")
    print(f"Mean confidence: {mean_confidence:.3f}")
    print(f"OK readings: {ok_percentage:.1f}%")
    print(f"Number of readings: {len(hr)}")

# ==========================================
# COMBINED HR ANALYSIS
# ==========================================

combined = pd.to_numeric(
    data["Combined_HR"], errors="coerce"
).dropna()

if len(combined) > 0:

    print("\n--- COMBINED CHROM HR ---")
    print(f"Mean HR: {combined.mean():.2f} BPM")
    print(f"Standard deviation: {combined.std():.2f} BPM")
    print(f"Minimum HR: {combined.min():.2f} BPM")
    print(f"Maximum HR: {combined.max():.2f} BPM")
    print(f"Valid readings: {len(combined)}")

# ==========================================
# SAVE RESULTS
# ==========================================

summary = pd.DataFrame(results)

summary.to_csv(
    "chrom_stability_summary.csv",
    index=False
)

print("\n==========================================")
print("CHROM STABILITY ANALYSIS COMPLETED")
print("Saved: chrom_stability_summary.csv")
print("==========================================")
