# FaceVitals – Development Progress Log

**Project:** FaceVitals  
**Team:** SignalPioneers – IEEE Signal Processing Cup 2027  
**Update date:** 24 November 2026  
**Status:** Integrated prototype; heart-rate accuracy is still under validation.

## 1. Work completed

- Integrated a Tkinter application interface for FaceVitals.
- Added webcam input.
- Added video-file upload for offline testing.
- Integrated MediaPipe Face Landmarker for face detection and facial-region tracking.
- Defined three facial regions of interest (ROIs): forehead, left cheek, and right cheek.
- Added RGB signal extraction from the facial regions.
- Integrated POS-based signal processing and heart-rate estimation.
- Integrated CHROM-based signal processing and heart-rate estimation.
- Added regional POS and CHROM results and confidence/status indicators.
- Added a combined result that is displayed only when POS and CHROM estimates meet the application's agreement and confidence checks.
- Added CSV export of recorded measurements and signal-related results.
- Corrected the MediaPipe video timestamp handling so timestamps are intended to increase monotonically across frames and source sessions.

## 2. Current application outputs

The application is designed to show:

- POS heart-rate estimate.
- CHROM heart-rate estimate.
- Regional estimates for forehead, left cheek, and right cheek.
- Confidence/status information.
- Final combined BPM when the POS and CHROM estimates pass the application's checks.
- CSV output for later analysis.

## 3. Validation observation – important limitation

A recent test was performed using a video with a reference pulse-rate value of **104 PR**. FaceVitals displayed **60.4 BPM**.

- Reference: 104 PR
- FaceVitals result: 60.4 BPM
- Difference: 43.6 BPM lower than the reference

This is a significant mismatch. It means the current output must **not** be treated as a validated or reliable heart-rate measurement.

The application is currently an engineering prototype. The result from this single test does not establish the overall performance of the system, but it clearly shows that more algorithm debugging and validation are required.

## 4. Work still required

1. Verify that the reference pulse-rate value and video recording correspond to the same time interval.
2. Inspect the extracted RGB traces from forehead and cheeks for motion, lighting changes, and weak skin-signal quality.
3. Review POS and CHROM implementation details, including normalization, filtering, windowing, and sampling frequency.
4. Review the heart-rate frequency selection and confidence calculation to identify possible wrong-peak selection.
5. Compare POS, CHROM, and combined estimates against a pulse oximeter during synchronized recordings.
6. Record several tests under controlled lighting and with minimal head movement.
7. Only describe the system as validated after sufficient reference comparisons show acceptable agreement.

## 5. Important notes for the team

- A visible BPM value does not automatically mean that the value is accurate.
- The combined BPM is an algorithmic estimate, not a medical measurement.
- Confidence/status indicators are software quality indicators and are not yet clinically calibrated.
- Do not use FaceVitals for diagnosis or medical decisions.
- Keep the validation results, including failed tests, in the project records.

## 6. Current status summary

**Application integration:** Implemented in the current prototype.  
**Webcam and video input:** Implemented; runtime testing should continue.  
**POS and CHROM outputs:** Integrated; numerical accuracy remains under investigation.  
**Combined BPM:** Implemented with software checks; not yet validated.  
**CSV export:** Implemented in the application.  
**Reference validation:** In progress; a significant mismatch has been observed.

