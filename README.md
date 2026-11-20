# FaceVitals
### Contactless and Bias-Robust Vital Signs Estimation Using RGB Facial Videos

**Project Team:** SignalPioneers  
**Competition:** IEEE 2027 Signal Processing Cup

## 1. Project Overview

FaceVitals is a research and engineering project that explores contactless estimation of human vital signs using ordinary RGB facial videos captured by a standard camera.

The project investigates how subtle color variations in facial skin, caused by blood volume changes, can be processed to estimate physiological information without requiring physical contact with the person.

Our initial development focuses on heart rate estimation using remote photoplethysmography (rPPG).

## 2. Project Objectives

The main objectives are to:

- Estimate heart rate from facial RGB video.
- Extract facial regions of interest (ROIs).
- Process RGB signals using signal processing techniques.
- Implement and compare POS and CHROM algorithms.
- Improve signal quality and estimation stability.
- Explore robustness under different recording conditions.

## 3. System Architecture

The current prototype follows this processing pipeline:

1. Video acquisition using a webcam.
2. Face detection and tracking.
3. Extraction of RGB signals from facial regions.
4. Signal processing using POS and CHROM.
5. Bandpass filtering.
6. Heart rate estimation using FFT.
7. Signal quality assessment and result display.

## 4. Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Pandas
- Matplotlib
- Tkinter
- Visual Studio Code

## 5. Signal Processing Methods

### POS (Plane-Orthogonal-to-Skin)

POS is implemented to extract pulse-related information from normalized RGB signals while reducing the influence of certain illumination variations.

### CHROM (Chrominance-Based Method)

CHROM is implemented as a second rPPG method to extract pulse-related signals using chrominance information.

Both methods are evaluated and compared using the same recorded RGB data.

## 6. Current Development Status

The current prototype includes:

- Webcam-based video acquisition.
- Facial detection using MediaPipe.
- RGB signal extraction from multiple facial regions.
- POS and CHROM signal processing.
- Bandpass filtering.
- FFT-based heart rate estimation.
- Signal quality and stability analysis.
- A live desktop application for heart rate monitoring.

**Current scope:** Heart rate estimation is the implemented stage. SpO₂ and blood pressure estimation remain future development objectives and have not yet been validated.

## 7. Installation and Usage

### Requirements

Before running FaceVitals, make sure you have:

- Windows 10/11 or another supported operating system.
- Python 3 installed (use a version compatible with the packages in `requirements.txt`).
- Git installed.
- A working webcam.
- The MediaPipe Face Landmarker model file: `face_landmarker.task`.

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/SignalPioneers/FaceVitals.git
```

Move into the project directory:

```bash
cd FaceVitals
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Check the MediaPipe Model

Ensure that `face_landmarker.task` is available in the project root directory, where the application expects it.

### Step 5: Run the Application

```bash
python facevitals_app.py
```

Click the **START** button to activate the webcam and begin real-time heart rate estimation. Allow the system time to collect video frames before observing the estimated heart rate and signal quality indicators.

Click **STOP** to stop the measurement.

### Troubleshooting

- If a required Python module is missing, activate the virtual environment and reinstall dependencies.
- If the webcam does not start, check camera permissions and ensure that no other application is using it.
- If the MediaPipe model cannot be found, confirm that `face_landmarker.task` is in the expected project directory.


## 8. Project Structure

The repository contains Python scripts for:

- Webcam and face detection.
- RGB signal extraction.
- Signal resampling.
- POS and CHROM algorithms.
- Filtering and heart rate estimation.
- Signal quality analysis.
- Stability analysis and algorithm comparison.
- Live application development.

## 9. Project Documentation

For contributors, researchers and anyone interested in understanding the implementation, the following documentation is available:

- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)** – Explains the project pipeline, processing stages and how the different scripts work together.
- **[Python Code Documentation](docs/CODE_DOCUMENTATION.md)** – Describes the purpose, inputs and outputs of the project's Python files.

## 10. Limitations

The current prototype is a research demonstration, not a certified medical device.

Its estimates can be affected by lighting, movement, camera frame rate, skin appearance, and other recording conditions.

Independent reference-device validation is required before making clinical accuracy claims.

## 11. Future Development

Planned development includes:

- Improving robustness across different recording conditions and skin tones.
- More comprehensive testing with independent reference measurements.
- Further investigation of SpO₂ estimation.
- Research into blood pressure estimation.
- Exploring mobile deployment.

## 12. Team

**SignalPioneers**

This project is developed as part of the IEEE 2027 Signal Processing Cup.

---

*Building contactless health-monitoring technology through signal processing and engineering.*
