# FaceVitals – Development Progress Report

**Date:** 20 November 2026  
**Team:** SignalPioneers – IEEE 2027 Signal Processing Cup  
**Stage:** Baseline POS Offline Video Processing

## 1. Completed Work

The team has successfully implemented and executed an offline video-processing pipeline using Python, OpenCV, MediaPipe and the POS algorithm.

The new script `extract_rgb_from_video.py` processes recorded video files without requiring a live webcam.

## 2. Baseline Video

- File: `baseline.mp4`
- Resolution: 1920 × 1080
- Frame rate: approximately 29.99 FPS
- Duration: 120.06 seconds
- Total frames processed: 3,601
- Face detection rate: 100%
- Valid RGB extraction rate: 100%

## 3. Processing Results

| Processing stage | Result |
|---|---|
| Offline RGB extraction | Successful |
| RGB resampling | Successful at 30 Hz |
| POS processing | Successful for all three ROIs |
| Bandpass filtering | Successful |
| Heart-rate estimation | Successful |

### POS Heart Rate Results

| Region | Smoothed HR | Confidence | Status |
|---|---:|---:|---|
| Forehead | 58.89 BPM | 0.487 | OK |
| Left Cheek | 59.77 BPM | 0.297 | Low confidence |
| Right Cheek | 55.37 BPM | 0.486 | OK |

**Combined POS HR: 57.13 BPM (status: OK).**

The system generated 111 rolling heart-rate estimates.

Confidence is a spectral quality score, not a probability of correctness. These results have not yet been validated against a synchronized reference pulse oximeter or ECG.

## 4. Generated Outputs

- `rgb_signals.csv`
- `rgb_signals_resampled.csv`
- `pos_signals.csv`
- `pos_signals_complete.png`
- `filtered_signals.csv`
- `filtered_signals.png`
- `heart_rate_results.csv`
- `heart_rate_spectrum.png`

## 5. Next Activities

1. Preserve the baseline results.
2. Process `C1_low_50_v2.mj2` using the offline pipeline.
3. Process the other C1 brightness conditions.
4. Compare heart-rate estimates and signal quality against baseline.
5. Obtain synchronized reference measurements for validation.

**Important experimental note:** The current C1 low-light videos were generated through digital brightness attenuation. They are simulated low-light stress-test videos, not recordings verified at a physical illuminance of 50 lux or below.

## 6. Current Project Status

The baseline offline POS pipeline is operational. The next planned activity is the C1 50% brightness experiment.