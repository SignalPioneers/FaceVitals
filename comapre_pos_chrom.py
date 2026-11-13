
import numpy as np
import pandas as pd
import os

# ==========================================
# FACEVITALS - POS vs CHROM COMPARISON
# ==========================================

POS_FILE = "heart_rate_results.csv"
CHROM_FILE = "chrom_heart_rate_results.csv"

for file in [POS_FILE, CHROM_FILE]:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"{file} not found. Run both HR estimation scripts first."
        )

pos = pd.read_csv(POS_FILE)
chrom = pd.read_csv(CHROM_FILE)

regions = ["Forehead", "Left Cheek", "Right Cheek"]

print("\n==========================================")
print("       POS vs CHROM COMPARISON")
print("==========================================")

# ==========================================
# 1. CHECK REQUIRED COLUMNS
# ==========================================

required = ["Time", "Combined_HR", "Combined_Status"]

for region in regions:
    required.extend([
        f"{region}_Smoothed_HR",
        f"{region}_Confidence",
        f"{region}_Status"
    ])

for name, df in [("POS", pos), ("CHROM", chrom)]:
    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"{name} file is missing column: {column}"
            )

# ==========================================
# 2. REGIONAL PERFORMANCE
# ==========================================

print("\n========== REGIONAL PERFORMANCE ==========")

for region in regions:

    print(f"\n--- {region.upper()} ---")

    for name, df in [("POS", pos), ("CHROM", chrom)]:

        hr = df[f"{region}_Smoothed_HR"].dropna()
        confidence = df[f"{region}_Confidence"].dropna()

        total = len(df)
        ok_count = (
            df[f"{region}_Status"] == "ok"
        ).sum()

        print(f"\n{name}:")

        print(
            f"Mean HR: {hr.mean():.2f} BPM"
            if len(hr) else "Mean HR: unavailable"
        )

        print(
            f"HR standard deviation: {hr.std():.2f} BPM"
            if len(hr) > 1 else "HR standard deviation: unavailable"
        )

        print(
            f"Mean confidence: {confidence.mean():.3f}"
            if len(confidence) else "Mean confidence: unavailable"
        )

        print(
            f"OK readings: {ok_count}/{total} "
            f"({ok_count/total*100:.1f}%)"
            if total else "OK readings: unavailable"
        )

# ==========================================
# 3. COMBINED HR COMPARISON
# ==========================================

print("\n========== COMBINED HR COMPARISON ==========")

for name, df in [("POS", pos), ("CHROM", chrom)]:

    valid = df["Combined_HR"].dropna()

    print(f"\n{name}:")

    if len(valid) > 0:

        print(f"Mean: {valid.mean():.2f} BPM")
        print(f"Median: {valid.median():.2f} BPM")
        print(f"Standard deviation: {valid.std():.2f} BPM")
        print(f"Minimum: {valid.min():.2f} BPM")
        print(f"Maximum: {valid.max():.2f} BPM")
        print(f"Valid readings: {len(valid)}/{len(df)}")

    else:
        print("No valid combined HR readings.")

# ==========================================
# 4. MATCH BOTH ALGORITHMS BY TIME
# ==========================================

paired = pd.merge(
    pos[["Time", "Combined_HR"]],
    chrom[["Time", "Combined_HR"]],
    on="Time",
    how="inner",
    suffixes=("_POS", "_CHROM")
)

paired = paired.dropna(
    subset=["Combined_HR_POS", "Combined_HR_CHROM"]
)

print("\n========== PAIRED COMPARISON ==========")

print("Matching valid time points:", len(paired))

if len(paired) > 0:

    paired["Difference"] = (
        paired["Combined_HR_POS"] -
        paired["Combined_HR_CHROM"]
    )

    paired["Absolute_Difference"] = (
        paired["Difference"].abs()
    )

    print(
        "Mean POS - CHROM difference:",
        f"{paired['Difference'].mean():.2f} BPM"
    )

    print(
        "Mean absolute POS-CHROM difference:",
        f"{paired['Absolute_Difference'].mean():.2f} BPM"
    )

    print(
        "Maximum absolute difference:",
        f"{paired['Absolute_Difference'].max():.2f} BPM"
    )

    if len(paired) >= 2:
        correlation = paired[
            ["Combined_HR_POS", "Combined_HR_CHROM"]
        ].corr().iloc[0, 1]

        print(
            "Correlation between POS and CHROM:",
            f"{correlation:.3f}"
        )

    else:
        print("Correlation unavailable: fewer than 2 paired points.")

else:
    print("No matching valid time points.")

# ==========================================
# 5. SAVE COMPARISON RESULTS
# ==========================================

paired.to_csv(
    "pos_chrom_paired_comparison.csv",
    index=False
)

print("\nPaired comparison saved to:")
print("pos_chrom_paired_comparison.csv")

print("\n==========================================")
print("POS vs CHROM COMPARISON COMPLETED!")
print("==========================================")

print("\nIMPORTANT:")
print("Agreement between POS and CHROM does not prove accuracy.")
print("A synchronized reference device is needed for validation.")
