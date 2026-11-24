# Team Update – FaceVitals

Hi team,

I have updated the FaceVitals application and pushed the latest development progress.

### What has been added
- Integrated webcam and video-upload input.
- MediaPipe face detection with forehead and cheek ROIs.
- POS and CHROM heart-rate estimates.
- Regional results, confidence/status indicators, and a final combined BPM output.
- CSV export.
- A progress log documenting the current implementation and validation status.

### Important validation finding
In one recent video test, the reference pulse rate was 104 PR while FaceVitals displayed 60.4 BPM. This is a significant mismatch, so the current heart-rate output is not yet reliable or validated.

Please review the latest `facevitals_app.py` and `docs/PROGRESS_LOG.md`. The next priority is to investigate POS/CHROM signal processing and heart-rate peak selection, then validate against synchronized pulse-oximeter readings.

The application is an integrated prototype, not a clinically validated measurement tool.

Thank you.
