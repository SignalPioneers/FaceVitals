# FaceVitals – Development Progress Log

**Project:** FaceVitals  
**Team:** SignalPioneers – IEEE Signal Processing Cup 2027  
**Update date:** 25 November 2026  
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



## 7. Code corrections – 25 November 2026

- Updated the live application's cheek sampling to use landmark-guided rectangular regions that exclude the central nose area. The forehead remains a separate face-box ROI. The regions are not polygon masks; the ROI geometry and skin-mask quality still require validation with saved webcam footage.
- Kept the forehead region as a separate conservative face-box ROI.
- Added an autocorrelation periodicity check to the live app, POS offline heart-rate estimator, and CHROM offline heart-rate estimator.
- When the dominant FFT estimate and the autocorrelation period disagree by more than 12 BPM, the estimate is rejected / marked low-confidence instead of being accepted into the rolling median. The live application displays an ambiguous-signal status rather than publishing that BPM.
- A basic synthetic-signal check at 60–180 BPM showed close FFT/autocorrelation agreement for clean single-frequency signals. This is only a software sanity check; it does not demonstrate real-camera accuracy.

**Important:** These changes reduce two known sources of error but do not prove that the 60.4 BPM versus 104 PR mismatch is solved. Motion artefacts, lighting variation, ROI placement, and peak-selection ambiguity can still produce an incorrect estimate. The next required test is a synchronized recording with a pulse oximeter, checking the same time interval.


## 8. Live webcam and signal-pipeline corrections – 25 November 2026

The integrated application was updated after a synchronized pulse-oximeter comparison showed a substantial difference (73 PR versus 55.9 BPM).

- Added timestamp-based RGB resampling onto the configured uniform analysis clock. Webcam frames are no longer assumed to arrive at exactly 30 FPS; short frame intervals are interpolated, while long gaps reset the affected region's signal state rather than stitching across a dropout.
- Changed live CHROM processing to overlap-add every 1.6-second window, matching the project's offline CHROM implementation instead of emitting only the center sample from each window.
- Added a minimum skin-pixel ratio of 10% for each ROI. The webcam preview now shows each ROI in green when the skin-pixel requirement is met and red when it is not, together with the measured skin percentage.
- Added the analysis sampling frequency, per-region skin ratios, raw RGB channels, and frame-to-frame RGB change percentages to the exported CSV. The webcam preview shows the skin-pixel ratio and short-term RGB change for each ROI; the RGB-change percentage is a diagnostic indicator, not a pulse-quality probability.
- Tightened result gates: POS and CHROM each require at least two valid facial regions that agree within the configured spread. A final combined BPM is shown only when both algorithm-level estimates are available and differ by no more than the configured agreement threshold. Single-region or disagreeing outputs are not presented as a valid combined BPM.
- Requested 1280×720 at 30 FPS from the webcam when supported; timestamp resampling remains responsible for the actual analysis timing. Uploaded videos are now restricted to 20–60 FPS so the 1.6-second CHROM filtering window has sufficient samples.

**Verification status:** The changes are committed to the feature branch, but the updated application has not yet been executed on the user's computer or revalidated against the pulse oximeter. Python compilation, a new synchronized recording, and inspection of the exported CSV are still required. No accuracy claim is made from these code changes alone.
