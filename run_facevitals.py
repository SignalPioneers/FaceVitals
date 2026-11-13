
import subprocess
import sys
from pathlib import Path
import time

# =====================================================
# FACEVITALS - ONE CLICK PIPELINE
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

PIPELINE = [
    ("RGB VIDEO CAPTURE", "rgb_extraction.py"),
    ("RGB RESAMPLING", "resample_rgb.py"),
    ("RGB SIGNAL PLOTTING", "plot_rgb_signals.py"),
    ("POS ALGORITHM", "pos_algoritm.py"),
    ("BANDPASS FILTERING", "filtering.py"),
    ("HEART RATE ESTIMATION", "heart_rate.py"),
    ("SIGNAL QUALITY ASSESSMENT", "signal_quality.py")
]


def run_stage(stage_name, filename):

    print("\n" + "=" * 60)
    print(f"STARTING: {stage_name}")
    print("=" * 60)

    script_path = PROJECT_DIR / filename

    if not script_path.exists():
        raise FileNotFoundError(
            f"Cannot find required file: {filename}"
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed. "
            f"Check the error above before continuing."
        )

    print(f"\nCOMPLETED: {stage_name}")


def main():

    start_time = time.time()

    print("\n")
    print("*" * 60)
    print("          FACEVITALS ONE-CLICK SYSTEM")
    print("*" * 60)

    print("\nThis system will:")
    print("1. Capture facial video.")
    print("2. Extract and resample RGB signals.")
    print("3. Apply POS processing.")
    print("4. Filter the signals.")
    print("5. Estimate heart rate.")
    print("6. Assess signal quality.")

    input("\nPress ENTER to start FaceVitals...")

    try:

        for stage_name, filename in PIPELINE:

            run_stage(stage_name, filename)

        elapsed_time = time.time() - start_time

        print("\n")
        print("*" * 60)
        print("       FACEVITALS PIPELINE COMPLETED!")
        print("*" * 60)

        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

        print("\nGenerated results:")
        print("- rgb_signals.csv")
        print("- rgb_signals_resampled.csv")
        print("- pos_signals.csv")
        print("- filtered_signals.csv")
        print("- heart_rate_results.csv")
        print("- signal_quality_results.csv")

        print("\nFaceVitals processing finished successfully.")

    except Exception as error:

        print("\n")
        print("!" * 60)
        print("       FACEVITALS PIPELINE STOPPED")
        print("!" * 60)

        print(f"\nReason: {error}")

        print("\nCorrect the problem and run the system again.")


if __name__ == "__main__":
    main()
