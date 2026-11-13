# FaceVitals Python Code Documentation

This inventory is based on inspection of the 24 Python files supplied in the project ZIP. It describes their current observed roles; it does not imply that every script is required to run the application.

## A. Camera, face detection and RGB extraction

| File | Role | Input | Output / result |
|---|---|---|---|
| `webcam.py` | Minimal OpenCV webcam preview and camera access check | Camera device 0 | Live preview window |
| `camera_test.py` | Short webcam availability test | Camera device 0 | Camera-open/read status |
| `face_detection.py` | Standalone MediaPipe Face Landmarker demonstration; draws facial landmarks | Webcam frames; `face_landmarker.task` | Annotated live preview |
| `roi_extraction.py` | Standalone demonstration of face bounding box and forehead/cheek ROI placement | Webcam frames; `face_landmarker.task` | Annotated live preview |
| `rgb_extraction.py` | Main offline RGB acquisition script; detects face, defines three rectangular ROIs, applies HSV skin mask, computes mean RGB and records timestamps | Webcam frames; `face_landmarker.task` | `rgb_signals.csv` and preview windows |

`rgb_signals.csv` contains a time column and RGB channels for the forehead, left cheek and right cheek. The current HSV thresholds are experimental heuristics, not a universal skin detector.

## B. Signal preparation and visualization

| File | Role | Input | Output / result |
|---|---|---|---|
| `check_timestamps.py` | Checks timestamp values and sampling intervals | `rgb_signals.csv` | Diagnostic terminal output |
| `check_segments.py` | Finds gaps and continuous segments in the original RGB recording | `rgb_signals.csv` | Diagnostic terminal output |
| `resample_rgb.py` | Cleans timestamp/RGB rows, finds the longest continuous segment, resamples to uniform 30 Hz | `rgb_signals.csv` | `rgb_signals_resampled.csv` |
| `plot_rgb_signals.py` | Plots forehead and cheek RGB channel traces | `rgb_signals.csv` | Matplotlib figures |
| `check_columns.py` | Checks/displays the column structure of the POS CSV | `pos_signals.csv` | Diagnostic terminal output |

## C. POS processing and heart-rate estimation

| File | Role | Input | Output / result |
|---|---|---|---|
| `pos_algoritm.py` | Computes POS pulse signals using overlapping windows and overlap-add for each facial region | `rgb_signals_resampled.csv` | `pos_signals.csv`, `pos_signals_complete.png` |
| `filtering.py` | Applies Butterworth band-pass filtering to the POS signals, while checking timestamps and continuous valid samples | `pos_signals.csv` | `filtered_signals.csv`, `filtered_signals.png` |
| `heart_rate.py` | Estimates rolling HR from filtered POS signals; computes regional and combined estimates, confidence/status and plots spectra | `filtered_signals.csv` | `heart_rate_results.csv`, `heart_rate_spectrum.png` |
| `signal_quality.py` | Calculates POS signal-quality metrics over analysis windows | `filtered_signals.csv` | `signal_quality_results.csv`, `signal_quality_scores.png` |
| `pos_stability_analysis.py` | Summarizes variability and status of the POS HR estimates | `heart_rate_results.csv` | `pos_stability_summary.csv` |

## D. CHROM processing and heart-rate estimation

| File | Role | Input | Output / result |
|---|---|---|---|
| `chrom_algorithm.py` | Computes CHROM signals from normalized RGB using X/Y projections, filtering and overlapping-window processing | `rgb_signals_resampled.csv` | `chrom_signals.csv` and a CHROM plot |
| `chrom_filtering.py` | Processes the saved CHROM signals into the file used by downstream CHROM analysis | `chrom_signals.csv` | `chrom_filtered_signals.csv` and a plot |
| `chrom_heart_rate.py` | Estimates rolling HR and spectral confidence from CHROM signals | `chrom_filtered_signals.csv` | `chrom_heart_rate_results.csv` and a spectrum plot |
| `chrom_signal_quality.py` | Calculates CHROM signal-quality metrics | `chrom_filtered_signals.csv` | `chrom_signal_quality_results.csv`, `chrom_signal_quality_scores.png` |
| `chrom_stability_analysis.py` | Summarizes stability of CHROM HR estimates | `chrom_heart_rate_results.csv` | CHROM stability summary CSV |

## E. POS-vs-CHROM comparison

| File | Role | Input | Output / result |
|---|---|---|---|
| `comapre_pos_chrom.py` | Earlier/narrower comparison of POS and CHROM HR result files | `heart_rate_results.csv`, `chrom_heart_rate_results.csv` | Paired comparison CSV and terminal summaries |
| `compare_pos_chrom.py` | More complete comparison of HR and signal-quality outputs, including plots | POS/CHROM HR result CSVs and POS/CHROM quality result CSVs | `pos_chrom_comparison.csv`, `pos_chrom_hr_comparison.png`, `pos_chrom_quality_comparison.png` |

The two comparison scripts are **not identical duplicates**. `compare_pos_chrom.py` is longer and includes quality-result comparison and plots. `comapre_pos_chrom.py` is a narrower HR-only comparison and its filename is misspelled (`comapre` rather than `compare`). Do not delete either file until the team confirms whether the older script is still needed.

## F. Integrated live app and launcher

| File | Role | Input | Output / result |
|---|---|---|---|
| `facevitals_app.py` | Integrated Tkinter live application: camera, MediaPipe face tracking, RGB ROI extraction, live POS, causal filtering, HR/confidence display and CSV recording | Webcam; `face_landmarker.task` | Live UI and a timestamped CSV recording |
| `run_facevitals.py` | Sequential launcher for the offline POS workflow | Calls other scripts in sequence | Runs RGB capture → resampling → RGB plotting → POS → filtering → POS HR → POS quality |

## G. Main third-party libraries observed

- `opencv-python` (`cv2`): webcam frames and image processing.
- `mediapipe`: face landmark detection.
- `numpy`: numerical arrays and signal calculations.
- `pandas`: CSV input/output and tabular data.
- `scipy`: Butterworth filters and signal-processing utilities.
- `matplotlib`: signal and result plots.
- `tkinter`: desktop application interface.
- `Pillow` (`PIL`): display camera frames in Tkinter.
- Python standard library modules such as `csv`, `os`, `time`, `datetime`, `subprocess`, `pathlib`, `collections` and `sys`.

## H. Recommended execution sequence

### POS offline
`rgb_extraction.py` → `resample_rgb.py` → `pos_algoritm.py` → `filtering.py` → `heart_rate.py` → `signal_quality.py`

Optional RGB inspection: `plot_rgb_signals.py` after RGB capture. The supplied launcher `run_facevitals.py` also includes this plotting stage.

### CHROM offline
`rgb_extraction.py` → `resample_rgb.py` → `chrom_algorithm.py` → `chrom_filtering.py` → `chrom_heart_rate.py` → `chrom_signal_quality.py`

### Evaluation
Run `pos_stability_analysis.py` and `chrom_stability_analysis.py` after the respective HR outputs exist. Run `compare_pos_chrom.py` after both HR and both quality result CSVs exist.

### Live demonstration
`facevitals_app.py` is run independently. It does not require running the offline scripts first, but it does require the MediaPipe model file and a working camera.

## I. Review points before publication

- Align or explicitly document the 0.7–3.5 Hz HR-estimation band versus the 0.7–3.0 Hz band in the two signal-quality scripts.
- Review the short-window filtering in `chrom_algorithm.py` and the role of `chrom_filtering.py` as a separate downstream step.
- Document that the current offline RGB extraction uses rectangular ROIs and HSV skin masking, with known robustness limitations.
- Make sure the README clearly distinguishes a real-time demonstration from validated vital-sign measurement.
- Validate HR estimates against synchronized independent reference measurements before making accuracy claims.
