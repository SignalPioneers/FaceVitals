
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FACEVITALS: POS VS CHROM COMPARATIVE ANALYSIS
# ============================================================

POS_HR_FILE = "heart_rate_results.csv"
CHROM_HR_FILE = "chrom_heart_rate_results.csv"

POS_QUALITY_FILE = "signal_quality_results.csv"
CHROM_QUALITY_FILE = "chrom_signal_quality_results.csv"

OUTPUT_FILE = "pos_chrom_comparison.csv"

HR_PLOT_FILE = "pos_chrom_hr_comparison.png"
QUALITY_PLOT_FILE = "pos_chrom_quality_comparison.png"


# ============================================================
# 1. LOAD HEART RATE RESULTS
# ============================================================

pos_hr = pd.read_csv(POS_HR_FILE)
chrom_hr = pd.read_csv(CHROM_HR_FILE)

required_hr_columns = [
    "Region",
    "Heart_Rate_BPM"
]

for column in required_hr_columns:

    if column not in pos_hr.columns:
        raise ValueError(
            f"Missing column in POS HR file: {column}"
        )

    if column not in chrom_hr.columns:
        raise ValueError(
            f"Missing column in CHROM HR file: {column}"
        )


# ============================================================
# 2. MERGE POS AND CHROM HEART RATE RESULTS
# ============================================================

hr_comparison = pd.merge(
    pos_hr[["Region", "Heart_Rate_BPM"]],
    chrom_hr[["Region", "Heart_Rate_BPM"]],
    on="Region",
    suffixes=("_POS", "_CHROM"),
    how="inner"
)

if hr_comparison.empty:
    raise ValueError(
        "No matching regions found in HR result files."
    )

hr_comparison["Absolute_Difference_BPM"] = (
    hr_comparison["Heart_Rate_BPM_POS"] -
    hr_comparison["Heart_Rate_BPM_CHROM"]
).abs()


# ============================================================
# 3. LOAD SIGNAL QUALITY RESULTS
# ============================================================

pos_quality = pd.read_csv(POS_QUALITY_FILE)
chrom_quality = pd.read_csv(CHROM_QUALITY_FILE)

required_quality_columns = [
    "Region",
    "Overall_Quality_Score"
]

for column in required_quality_columns:

    if column not in pos_quality.columns:
        raise ValueError(
            f"Missing column in POS quality file: {column}"
        )

    if column not in chrom_quality.columns:
        raise ValueError(
            f"Missing column in CHROM quality file: {column}"
        )


# ============================================================
# 4. MERGE QUALITY RESULTS
# ============================================================

quality_comparison = pd.merge(
    pos_quality[
        ["Region", "Overall_Quality_Score"]
    ],
    chrom_quality[
        ["Region", "Overall_Quality_Score"]
    ],
    on="Region",
    suffixes=("_POS", "_CHROM"),
    how="inner"
)

quality_comparison["Quality_Difference"] = (
    quality_comparison["Overall_Quality_Score_POS"] -
    quality_comparison["Overall_Quality_Score_CHROM"]
)


# ============================================================
# 5. COMBINE HEART RATE AND QUALITY RESULTS
# ============================================================

final_comparison = pd.merge(
    hr_comparison,
    quality_comparison,
    on="Region",
    how="left"
)

final_comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 6. DISPLAY HEART RATE COMPARISON
# ============================================================

print("\n==============================================")
print("       POS VS CHROM HEART RATE COMPARISON")
print("==============================================")

print(
    hr_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 7. DISPLAY QUALITY COMPARISON
# ============================================================

print("\n==============================================")
print("       POS VS CHROM QUALITY COMPARISON")
print("==============================================")

print(
    quality_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 8. IDENTIFY REGIONS WITH SMALLEST HR DIFFERENCE
# ============================================================

best_agreement = hr_comparison.loc[
    hr_comparison["Absolute_Difference_BPM"].idxmin()
]

largest_difference = hr_comparison.loc[
    hr_comparison["Absolute_Difference_BPM"].idxmax()
]

print("\n==============================================")
print("       COMPARISON SUMMARY")
print("==============================================")

print(
    f"Smallest HR difference: "
    f"{best_agreement['Region']} "
    f"({best_agreement['Absolute_Difference_BPM']:.2f} BPM)"
)

print(
    f"Largest HR difference: "
    f"{largest_difference['Region']} "
    f"({largest_difference['Absolute_Difference_BPM']:.2f} BPM)"
)

if not quality_comparison.empty:

    mean_pos_quality = quality_comparison[
        "Overall_Quality_Score_POS"
    ].mean()

    mean_chrom_quality = quality_comparison[
        "Overall_Quality_Score_CHROM"
    ].mean()

    print(
        f"Mean POS quality score: "
        f"{mean_pos_quality:.2f}/100"
    )

    print(
        f"Mean CHROM quality score: "
        f"{mean_chrom_quality:.2f}/100"
    )


# ============================================================
# 9. PLOT HEART RATE COMPARISON
# ============================================================

regions = hr_comparison["Region"]

x = range(len(regions))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    hr_comparison["Heart_Rate_BPM_POS"],
    width,
    label="POS"
)

plt.bar(
    [i + width / 2 for i in x],
    hr_comparison["Heart_Rate_BPM_CHROM"],
    width,
    label="CHROM"
)

plt.xticks(list(x), regions)

plt.xlabel("Facial Region")
plt.ylabel("Heart Rate (BPM)")

plt.title(
    "FaceVitals: POS vs CHROM Heart Rate Comparison"
)

plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(HR_PLOT_FILE, dpi=300)
plt.show()


# ============================================================
# 10. PLOT SIGNAL QUALITY COMPARISON
# ============================================================

quality_regions = quality_comparison["Region"]

x = range(len(quality_regions))

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    quality_comparison["Overall_Quality_Score_POS"],
    width,
    label="POS"
)

plt.bar(
    [i + width / 2 for i in x],
    quality_comparison["Overall_Quality_Score_CHROM"],
    width,
    label="CHROM"
)

plt.xticks(list(x), quality_regions)

plt.xlabel("Facial Region")
plt.ylabel("Overall Quality Score (0-100)")

plt.title(
    "FaceVitals: POS vs CHROM Signal Quality Comparison"
)

plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(QUALITY_PLOT_FILE, dpi=300)
plt.show()


# ============================================================
# 11. COMPLETION MESSAGE
# ============================================================

print("\n==============================================")
print("COMPARATIVE ANALYSIS COMPLETED!")
print("==============================================")

print(f"Comparison CSV saved: {OUTPUT_FILE}")
print(f"HR graph saved: {HR_PLOT_FILE}")
print(f"Quality graph saved: {QUALITY_PLOT_FILE}")
