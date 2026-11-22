
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# FACEVITALS - POS vs CHROM COMPARISON
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

POS_HR_FILE = PROJECT_DIR / "heart_rate_results.csv"
CHROM_HR_FILE = PROJECT_DIR / "chrom_heart_rate_results.csv"

POS_QUALITY_FILE = PROJECT_DIR / "signal_quality_results.csv"
CHROM_QUALITY_FILE = PROJECT_DIR / "chrom_signal_quality_results.csv"

OUTPUT_CSV = PROJECT_DIR / "pos_chrom_comparison.csv"

REGIONS = {
    "Forehead": "Forehead_Smoothed_HR",
    "Left Cheek": "Left Cheek_Smoothed_HR",
    "Right Cheek": "Right Cheek_Smoothed_HR",
    "Combined": "Combined_HR"
}


def main():

    print("\n========== FACEVITALS POS vs CHROM ==========\n")

    # Load files
    pos_hr = pd.read_csv(POS_HR_FILE)
    chrom_hr = pd.read_csv(CHROM_HR_FILE)

    pos_quality = pd.read_csv(POS_QUALITY_FILE)
    chrom_quality = pd.read_csv(CHROM_QUALITY_FILE)

    results = []

    # ==========================================
    # HEART RATE COMPARISON
    # ==========================================

    for region, column in REGIONS.items():

        pos_mean = pd.to_numeric(
            pos_hr[column], errors="coerce"
        ).mean()

        chrom_mean = pd.to_numeric(
            chrom_hr[column], errors="coerce"
        ).mean()

        difference = abs(pos_mean - chrom_mean)

        pos_confidence_column = region + "_Confidence"

        pos_confidence = (
            pos_hr[pos_confidence_column].mean()
            if pos_confidence_column in pos_hr.columns
            else None
        )

        chrom_confidence = (
            chrom_hr[pos_confidence_column].mean()
            if pos_confidence_column in chrom_hr.columns
            else None
        )

        # Quality scores are available for individual ROIs
        pos_quality_score = None
        chrom_quality_score = None

        if region != "Combined":

            pos_row = pos_quality[
                pos_quality["Region"] == region
            ]

            chrom_row = chrom_quality[
                chrom_quality["Region"] == region
            ]

            if not pos_row.empty:
                pos_quality_score = pos_row[
                    "Overall_Quality_Score"
                ].iloc[0]

            if not chrom_row.empty:
                chrom_quality_score = chrom_row[
                    "Overall_Quality_Score"
                ].iloc[0]

        results.append({
            "Region": region,
            "POS_Mean_HR_BPM": pos_mean,
            "CHROM_Mean_HR_BPM": chrom_mean,
            "HR_Difference_BPM": difference,
            "POS_Mean_Confidence": pos_confidence,
            "CHROM_Mean_Confidence": chrom_confidence,
            "POS_Quality_Score": pos_quality_score,
            "CHROM_Quality_Score": chrom_quality_score
        })

    comparison = pd.DataFrame(results)

    # Save comparison
    comparison.to_csv(OUTPUT_CSV, index=False)

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    print("POS vs CHROM RESULTS\n")
    print(comparison.round(3).to_string(index=False))

    print("\nMean HR difference:",
          round(comparison["HR_Difference_BPM"].mean(), 2),
          "BPM")

    print("\nMean POS quality:",
          round(pos_quality["Overall_Quality_Score"].mean(), 2))

    print("Mean CHROM quality:",
          round(chrom_quality["Overall_Quality_Score"].mean(), 2))

    # ==========================================
    # HEART RATE GRAPH
    # ==========================================

    hr_plot = comparison.set_index("Region")[
        ["POS_Mean_HR_BPM", "CHROM_Mean_HR_BPM"]
    ]

    hr_plot.plot(kind="bar", figsize=(10, 5))

    plt.title("POS vs CHROM Mean Heart Rate")
    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Facial Region")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        PROJECT_DIR / "pos_chrom_hr_comparison.png",
        dpi=300
    )

    plt.show()

    # ==========================================
    # SIGNAL QUALITY GRAPH
    # ==========================================

    quality_plot = comparison.set_index("Region")[
        ["POS_Quality_Score", "CHROM_Quality_Score"]
    ].dropna()

    quality_plot.plot(kind="bar", figsize=(9, 5))

    plt.title("POS vs CHROM Signal Quality")
    plt.ylabel("Overall Quality Score")
    plt.xlabel("Facial Region")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        PROJECT_DIR / "pos_chrom_quality_comparison.png",
        dpi=300
    )

    plt.show()

    print("\nComparison completed successfully.")
    print(f"CSV saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
