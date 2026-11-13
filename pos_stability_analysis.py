
import numpy as np
import pandas as pd
import os

# ==========================================
# FACEVITALS - POS STABILITY ANALYSIS
# ==========================================

INPUT_FILE = "heart_rate_results.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "heart_rate_results.csv not found. Run heart_rate.py first."
    )

data = pd.read_csv(INPUT_FILE)

regions = ["Forehead", "Left Cheek", "Right Cheek"]

print("\n==========================================")
print("       FACEVITALS POS STABILITY")
print("==========================================")

print("Total HR estimates:", len(data))

# ==========================================
# 1. ANALYSE EACH FACIAL REGION
# ==========================================

for region in regions:

    hr_column = f"{region}_Smoothed_HR"
    confidence_column = f"{region}_Confidence"
    status_column = f"{region}_Status"

    hr = data[hr_column].dropna()
    confidence = data[confidence_column].dropna()

    total = len(data)
    ok_count = (data[status_column] == "ok").sum()
    low_count = (data[status_column] == "low_confidence").sum()
    jump_count = (data[status_column] == "jump_warning").sum()

    print(f"\n--- {region.upper()} ---")

    if len(hr) > 0:

        print(f"Mean HR: {hr.mean():.2f} BPM")
        print(f"Median HR: {hr.median():.2f} BPM")
        print(f"Standard deviation: {hr.std():.2f} BPM")
        print(f"Minimum HR: {hr.min():.2f} BPM")
        print(f"Maximum HR: {hr.max():.2f} BPM")

    if len(confidence) > 0:
        print(f"Mean confidence: {confidence.mean():.3f}")

    print(f"OK readings: {ok_count}/{total} ({ok_count/total*100:.1f}%)")
    print(f"Low confidence: {low_count}/{total}")
    print(f"Jump warnings: {jump_count}/{total}")

# ==========================================
# 2. ANALYSE COMBINED POS HR
# ==========================================

combined = data["Combined_HR"].dropna()

valid_count = len(combined)

print("\n--- COMBINED POS HR ---")

if valid_count > 0:

    print(f"Mean HR: {combined.mean():.2f} BPM")
    print(f"Median HR: {combined.median():.2f} BPM")
    print(f"Standard deviation: {combined.std():.2f} BPM")
    print(f"Minimum HR: {combined.min():.2f} BPM")
    print(f"Maximum HR: {combined.max():.2f} BPM")
    print(f"Valid readings: {valid_count}/{len(data)}")

else:
    print("No valid combined HR readings.")

# ==========================================
# 3. SAVE SUMMARY
# ==========================================

summary_rows = []

for region in regions:

    hr = data[f"{region}_Smoothed_HR"].dropna()
    confidence = data[f"{region}_Confidence"].dropna()

    total = len(data)
    ok_count = (data[f"{region}_Status"] == "ok").sum()

    summary_rows.append({
        "Region": region,
        "Mean_HR": hr.mean() if len(hr) else np.nan,
        "Median_HR": hr.median() if len(hr) else np.nan,
        "Standard_Deviation": hr.std() if len(hr) else np.nan,
        "Minimum_HR": hr.min() if len(hr) else np.nan,
        "Maximum_HR": hr.max() if len(hr) else np.nan,
        "Mean_Confidence": confidence.mean() if len(confidence) else np.nan,
        "OK_Percentage": ok_count / total * 100 if total else 0
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv("pos_stability_summary.csv", index=False)

print("\nSummary saved to pos_stability_summary.csv")

print("\nPOS STABILITY ANALYSIS COMPLETED!")
