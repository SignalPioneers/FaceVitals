# FaceVitals
### Contactless Heart-Rate Estimation from RGB Facial Videos

**Project Team:** SignalPioneers  
**Competition:** IEEE 2027 Signal Processing Cup

## 1. Project Overview

FaceVitals is a research prototype for contactless heart-rate estimation from ordinary RGB facial videos. It analyses subtle changes in facial skin colour associated with pulse-related blood-volume changes.

The current application focuses on heart-rate estimation using remote photoplethysmography (rPPG). It is a research demonstration, not a certified medical device.

## 2. What the application does

The desktop application supports:
- Live webcam measurement.
- Uploading and processing a recorded video.
- Face detection and tracking.
- RGB extraction from the forehead, left cheek and right cheek.
- POS and CHROM rPPG processing.
- Heart-rate estimates in beats per minute (BPM), with signal-quality/status information.
- CSV export of recorded measurements.

POS and CHROM are independent methods. Their estimates may differ; the application should show quality/status information rather than force the values to match.

## 3. How it works

1. The user starts the webcam or selects a video file.
2. MediaPipe detects the face and facial landmarks.
3. FaceVitals extracts RGB colour information from selected facial regions.
4. POS and CHROM process the RGB changes to obtain pulse-related signals.
5. Signal filtering and frequency analysis estimate BPM.
6. The interface displays regional and algorithm results, a combined estimate when available, and quality/status information.

## 4. Quick start on Windows (recommended)

The easiest way to launch FaceVitals from the source repository is the included **START_FACEVITALS.bat** file.

### First-time setup

1. Install **Python 3** on Windows. During installation, enable **Add Python to PATH** if the installer offers that option.
2. Download the project using GitHub's **Code → Download ZIP**, then extract the ZIP. Alternatively, clone the repository with Git.
3. Make sure these files are together in the extracted project folder:
   - `START_FACEVITALS.bat`
   - `facevitals_app.py`
   - `face_landmarker.task`
   - `requirements.txt`
4. Double-click **START_FACEVITALS.bat**.
5. On the first run, the launcher creates a Python virtual environment and installs the required packages. This needs an internet connection and may take several minutes.
6. When installation finishes, FaceVitals opens. Click **UPLOAD VIDEO** to choose a recorded video, or **START WEBCAM** to use the camera.

### Later launches

After the first successful setup, simply double-click **START_FACEVITALS.bat** again. The existing environment is reused; you do not need to open VS Code or type commands.

### Important

- Keep `face_landmarker.task` in the same folder as `facevitals_app.py`. This model file is included in the repository.
- Do not move the launcher, application script, model file or requirements file out of the project folder.
- The first launch requires internet access to install Python packages.
- Windows may show a security prompt for a downloaded ZIP or batch file. Only run files obtained from the official project repository.

## 5. Manual setup (alternative)

If you prefer using PowerShell in the project folder:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python facevitals_app.py
```

If the `py` command is unavailable, use `python` instead, provided Python 3 is installed and available on PATH.

## 6. Using the application

- **UPLOAD VIDEO:** choose a supported video file from your computer and let FaceVitals process it.
- **START WEBCAM:** begin a live webcam session.
- **STOP:** stop the current source/session.
- **SAVE CSV:** save the measurement records generated during the session.

For better signal quality, use steady lighting, keep the face visible and reasonably still, and allow sufficient time for the signal window to fill. A preliminary estimate is not a confirmed measurement. If the signal is insufficient, a reliable BPM may not be available.

## 7. Technologies

- Python and Tkinter
- OpenCV
- MediaPipe
- NumPy and SciPy
- Pandas and Matplotlib
- Pillow

## 8. Troubleshooting

**The launcher says Python was not found**
- Install Python 3 and enable **Add Python to PATH**.
- Close and reopen the terminal or rerun the launcher after installation.

**The first launch cannot install packages**
- Check the internet connection.
- Open PowerShell in the project folder and run:
  `python -m pip install -r requirements.txt`.

**The application says `face_landmarker.task` is missing**
- Confirm that the model file is beside `facevitals_app.py` in the project folder. Re-download the full repository if it is missing.

**The webcam does not start**
- Allow camera access in Windows privacy settings.
- Close other applications that may be using the webcam.

**Upload Video is not visible**
- Make sure you downloaded/extracted the current project version and launched it using `START_FACEVITALS.bat` from that same folder.
- Do not launch an older copy of `facevitals_app.py` from another folder.

## 9. Project structure and documentation

The repository includes the main desktop application, rPPG and signal-analysis scripts, the MediaPipe model asset, requirements, sample diagnostic outputs and supporting documentation.

Additional documentation:
- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)** – project pipeline and processing stages.
- **[Python Code Documentation](docs/CODE_DOCUMENTATION.md)** – purpose and usage of Python files.

## 10. Scope and limitations

The current implemented application scope is heart-rate estimation. SpO₂ and blood-pressure estimation are future research objectives and are not validated features.

Results can be affected by movement, lighting, camera quality, frame rate, face tracking and other recording conditions. Accuracy claims require comparison with a synchronized, independent reference heart-rate device. FaceVitals must not be used for diagnosis or treatment decisions.

## 11. Team

**SignalPioneers**  
IEEE 2027 Signal Processing Cup

---
*Building contactless health-monitoring technology through signal processing and engineering.*
