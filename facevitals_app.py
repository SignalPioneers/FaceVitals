import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from datetime import datetime
from collections import deque
from PIL import Image, ImageTk
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt


# ============================================================
# FACEVITALS - POS + CHROM LIVE / VIDEO APPLICATION
# ============================================================

FS_DEFAULT = 30.0
POS_WINDOW_SECONDS = 1.6
HR_WINDOW_SECONDS = 10.0
FFT_SIZE = 2048
LOW_HR = 0.7
HIGH_HR = 3.5
CONFIDENCE_THRESHOLD = 0.30
FILTER_ORDER = 4
UPDATE_INTERVAL_SECONDS = 1.0
MAX_ALGORITHM_DIFFERENCE = 10.0
MAX_HR_JUMP = 20.0

# Forehead is a compact upper-face rectangle. Cheeks are rectangular
# sampling regions positioned from eye, nose-tip and upper-lip landmarks.
# The inner cheek edges stay well outside the nose wings.
FOREHEAD_ROI = (0.30, 0.08, 0.70, 0.27)
CHEEK_INNER_MARGIN = 0.13
CHEEK_OUTER_MARGIN = 0.15
CHEEK_TOP_MARGIN = 0.025
CHEEK_BOTTOM_MARGIN = 0.075
REGIONS = ["Forehead", "Left Cheek", "Right Cheek"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "face_landmarker.task was not found in the same folder as facevitals_app.py"
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(landmarker_options)


# ============================================================
# SESSION STATE
# ============================================================

capture = None
source_mode = None                 # "camera" or "video"
source_running = False
source_fps = FS_DEFAULT
session_fs = FS_DEFAULT
session_start = 0.0
last_mp_timestamp = -1
video_frame_index = 0
video_path = None
record_rows = []
camera_times = deque(maxlen=60)

pos_window = int(round(FS_DEFAULT * POS_WINDOW_SECONDS))
hr_window = int(round(FS_DEFAULT * HR_WINDOW_SECONDS))
update_interval = int(round(FS_DEFAULT * UPDATE_INTERVAL_SECONDS))
pos_filter_sos = butter(
    FILTER_ORDER, [LOW_HR, HIGH_HR],
    btype="bandpass", fs=FS_DEFAULT, output="sos"
)


def configure_sampling(fs):
    """Configure filters and sample counts for the current source FPS."""
    global session_fs, pos_window, hr_window, update_interval, pos_filter_sos
    session_fs = float(fs)
    pos_window = max(24, int(round(session_fs * POS_WINDOW_SECONDS)))
    hr_window = max(100, int(round(session_fs * HR_WINDOW_SECONDS)))
    update_interval = max(1, int(round(session_fs * UPDATE_INTERVAL_SECONDS)))
    pos_filter_sos = butter(
        FILTER_ORDER, [LOW_HR, HIGH_HR],
        btype="bandpass", fs=session_fs, output="sos"
    )


def create_region_state():
    return {
        "rgb": deque(maxlen=pos_window),
        "overlap_sum": np.zeros(pos_window, dtype=float),
        "overlap_count": np.zeros(pos_window, dtype=float),
        "pos_filter_zi": None,
        "pos_filtered": deque(maxlen=hr_window),
        "chrom_rgb": deque(maxlen=pos_window),
        "chrom_filtered": deque(maxlen=hr_window),
        "chrom_sos": butter(
            FILTER_ORDER, [LOW_HR, HIGH_HR],
            btype="bandpass", fs=session_fs, output="sos"
        ),
        "pos_history": deque(maxlen=5),
        "chrom_history": deque(maxlen=5),
        "pos_hr": None,
        "chrom_hr": None,
        "pos_raw_hr": None,
        "chrom_raw_hr": None,
        "pos_confidence": 0.0,
        "chrom_confidence": 0.0,
        "pos_status": "Warming up",
        "chrom_status": "Warming up",
        "pos_previous_hr": None,
        "chrom_previous_hr": None,
        "pos_pending_hr": None,
        "chrom_pending_hr": None,
        "pos_pending_count": 0,
        "chrom_pending_count": 0,
        "pos_sample_count": 0,
        "chrom_sample_count": 0,
        "pos_last_update": 0,
        "chrom_last_update": 0,
        "pos_peak_diagnostics": "", "chrom_peak_diagnostics": "",
        "pos_autocorr_bpm": None, "chrom_autocorr_bpm": None,
    }


region_states = {}


def reset_all_regions():
    global region_states
    region_states = {name: create_region_state() for name in REGIONS}


configure_sampling(FS_DEFAULT)
reset_all_regions()


# ============================================================
# FACE / SKIN / RGB EXTRACTION
# ============================================================

def create_skin_mask(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 20, 40])
    upper1 = np.array([25, 200, 255])
    lower2 = np.array([160, 20, 40])
    upper2 = np.array([180, 200, 255])
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )


def extract_roi_rgb(rgb_frame, landmarks, bbox, region):
    """Extract skin RGB from the forehead or a nose-excluding cheek rectangle."""
    height, width = rgb_frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    face_width = x_max - x_min
    face_height = y_max - y_min

    if region == "Forehead":
        rx1, ry1, rx2, ry2 = FOREHEAD_ROI
        x1 = max(0, int(x_min + rx1 * face_width))
        y1 = max(0, int(y_min + ry1 * face_height))
        x2 = min(width, int(x_min + rx2 * face_width))
        y2 = min(height, int(y_min + ry2 * face_height))
    elif region in ("Left Cheek", "Right Cheek"):
        # FaceLandmarker indices: 1 = nose tip, 145/374 = lower eye
        # landmarks, 13 = upper lip. These landmarks adapt the ROIs to pose.
        required = (1, 13, 145, 374)
        if len(landmarks) <= max(required) or face_width <= 0 or face_height <= 0:
            return None, None, None

        nose_x = float(landmarks[1].x) * width
        lower_eye_y = (
            float(landmarks[145].y) + float(landmarks[374].y)
        ) * 0.5 * height
        upper_lip_y = float(landmarks[13].y) * height

        y1 = max(0, int(lower_eye_y + CHEEK_TOP_MARGIN * face_height))
        y2 = min(height, int(upper_lip_y - CHEEK_BOTTOM_MARGIN * face_height))

        if region == "Left Cheek":
            x1 = max(0, int(x_min + CHEEK_OUTER_MARGIN * face_width))
            x2 = min(width, int(nose_x - CHEEK_INNER_MARGIN * face_width))
        else:
            x1 = max(0, int(nose_x + CHEEK_INNER_MARGIN * face_width))
            x2 = min(width, int(x_max - CHEEK_OUTER_MARGIN * face_width))
    else:
        return None, None, None

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    roi_image = rgb_frame[y1:y2, x1:x2]
    if roi_image.size == 0:
        return None, (x1, y1, x2, y2), None

    region_mask = np.full(roi_image.shape[:2], 255, dtype=np.uint8)
    skin_mask = create_skin_mask(roi_image)
    combined_mask = cv2.bitwise_and(skin_mask, region_mask)
    selected = combined_mask > 0
    skin_pixels = roi_image[selected]

    if len(skin_pixels) < 20:
        return None, (x1, y1, x2, y2), None

    mean_rgb = np.mean(skin_pixels, axis=0)
    outline = np.array(
        [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
        dtype=np.int32
    )
    return mean_rgb, (x1, y1, x2, y2), outline

def process_pos(region, rgb_values):
    state = region_states[region]
    state["rgb"].append(np.asarray(rgb_values, dtype=float))

    if len(state["rgb"]) < pos_window:
        return

    rgb_window = np.asarray(state["rgb"], dtype=float)
    channel_means = np.mean(rgb_window, axis=0)
    if np.any(channel_means < 1e-8):
        return

    normalized = rgb_window / channel_means
    Rn, Gn, Bn = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    S1 = Gn - Bn
    S2 = Gn + Bn - 2.0 * Rn
    std_s2 = np.std(S2)
    if std_s2 < 1e-8:
        return

    alpha = np.std(S1) / std_s2
    h = S1 + alpha * S2
    h -= np.mean(h)

    state["overlap_sum"] += h
    state["overlap_count"] += 1.0
    count = state["overlap_count"][0]
    if count <= 0:
        return

    pos_sample = state["overlap_sum"][0] / count
    state["overlap_sum"][:-1] = state["overlap_sum"][1:]
    state["overlap_sum"][-1] = 0.0
    state["overlap_count"][:-1] = state["overlap_count"][1:]
    state["overlap_count"][-1] = 0.0

    if state["pos_filter_zi"] is None:
        state["pos_filter_zi"] = sosfilt_zi(pos_filter_sos) * pos_sample

    filtered, state["pos_filter_zi"] = sosfilt(
        pos_filter_sos, [pos_sample], zi=state["pos_filter_zi"]
    )
    state["pos_filtered"].append(float(filtered[0]))
    state["pos_sample_count"] += 1


# ============================================================
# CHROM STREAM PROCESSING
# ============================================================

def process_chrom(region, rgb_values):
    """
    CHROM is calculated over a rolling 1.6-second RGB window.
    X and Y are bandpass-filtered; the center sample is emitted
    to reduce the influence of the window edges.
    """
    state = region_states[region]
    state["chrom_rgb"].append(np.asarray(rgb_values, dtype=float))

    if len(state["chrom_rgb"]) < pos_window:
        return

    window = np.asarray(state["chrom_rgb"], dtype=float)
    means = np.mean(window, axis=0)
    if np.any(means < 1e-8):
        return

    normalized = window / means
    R, G, B = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + G - 1.5 * B

    try:
        Xf = sosfiltfilt(state["chrom_sos"], X)
        Yf = sosfiltfilt(state["chrom_sos"], Y)
    except ValueError:
        return

    std_y = np.std(Yf)
    if std_y < 1e-8:
        return

    alpha = np.std(Xf) / std_y
    chrom_signal = Xf - alpha * Yf
    chrom_signal -= np.mean(chrom_signal)

    center = len(chrom_signal) // 2
    state["chrom_filtered"].append(float(chrom_signal[center]))
    state["chrom_sample_count"] += 1


# ============================================================
# HEART RATE / CONFIDENCE
# ============================================================

def estimate_hr_from_signal(signal_values, history, previous_hr, pending_hr, pending_count):
    """Gate low-quality values and confirm sudden changes before publishing."""
    if len(signal_values) < hr_window:
        return None, 0.0, "Warming up", previous_hr, pending_hr, pending_count
    signal = np.asarray(signal_values, dtype=float)
    signal = signal - np.mean(signal)
    if not np.all(np.isfinite(signal)) or np.std(signal) < 1e-8:
        return None, 0.0, "Low signal", previous_hr, pending_hr, pending_count
    windowed = signal * np.hanning(hr_window)
    spectrum = np.fft.rfft(windowed, n=FFT_SIZE)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / session_fs)
    valid = (frequencies >= LOW_HR) & (frequencies <= HIGH_HR)
    if not np.any(valid):
        return None, 0.0, "No valid HR band", previous_hr, pending_hr, pending_count
    valid_indices = np.where(valid)[0]
    peak_index = valid_indices[np.argmax(power[valid])]
    peak_frequency = frequencies[peak_index]
    raw_hr = float(peak_frequency * 60.0)

    # Independent periodicity check: reject a spectral peak that disagrees
    # with the strongest plausible autocorrelation period.
    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    if autocorr[0] <= 1e-12:
        return None, 0.0, "Low signal", previous_hr, pending_hr, pending_count
    autocorr = autocorr / autocorr[0]
    min_lag = max(1, int(np.floor(session_fs / HIGH_HR)))
    max_lag = min(len(centered) - 1, int(np.ceil(session_fs / LOW_HR)))
    if max_lag <= min_lag:
        return None, 0.0, "Insufficient signal", previous_hr, pending_hr, pending_count
    lag_range = autocorr[min_lag:max_lag + 1]
    best_lag = min_lag + int(np.argmax(lag_range))
    autocorr_hr = float(60.0 * session_fs / best_lag)
    autocorr_quality = float(autocorr[best_lag])
    if autocorr_quality < 0.15 or abs(raw_hr - autocorr_hr) > 12.0:
        return None, min(0.29, max(0.0, autocorr_quality)), "Ambiguous HR signal", previous_hr, pending_hr, pending_count

    total_power = float(np.sum(power[valid]))
    if total_power <= 1e-12:
        return None, 0.0, "Low signal", previous_hr, pending_hr, pending_count
    fundamental = (np.abs(frequencies - peak_frequency) <= 0.1) & valid
    harmonic = (np.abs(frequencies - 2.0 * peak_frequency) <= 0.1) & valid
    confidence = float(np.clip(
        (np.sum(power[fundamental]) + np.sum(power[harmonic])) / total_power, 0.0, 1.0
    ))
    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence, "Low confidence", previous_hr, pending_hr, pending_count
    if previous_hr is not None and abs(raw_hr - previous_hr) > MAX_HR_JUMP:
        if pending_hr is not None and abs(raw_hr - pending_hr) <= 8.0:
            pending_count += 1
        else:
            pending_hr, pending_count = raw_hr, 1
        if pending_count < 2:
            return None, confidence, "Change pending", previous_hr, pending_hr, pending_count
        history.clear()
    else:
        pending_hr, pending_count = None, 0
    history.append(raw_hr)
    smoothed = float(np.median(history))
    return smoothed, confidence, "OK", smoothed, None, 0


def signal_diagnostics(signal_values):
    """Return strongest FFT candidates and strongest autocorrelation-period candidate."""
    x = np.asarray(signal_values, dtype=float)
    if len(x) < hr_window or not np.all(np.isfinite(x)):
        return "", None
    x = x - np.mean(x)
    if np.std(x) < 1e-8:
        return "", None
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / session_fs)
    band = np.where((freqs >= LOW_HR) & (freqs <= HIGH_HR))[0]
    ranked = band[np.argsort(spectrum[band])[-5:][::-1]]
    peaks = ";".join(f"{freqs[i]*60.0:.1f}:{spectrum[i]:.3g}" for i in ranked)
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    if ac[0] <= 1e-12:
        return peaks, None
    ac = ac / ac[0]
    lo = max(1, int(np.floor(session_fs / HIGH_HR)))
    hi = min(len(x)-1, int(np.ceil(session_fs / LOW_HR)))
    if hi <= lo:
        return peaks, None
    lag = lo + int(np.argmax(ac[lo:hi+1]))
    return peaks, float(60.0 * session_fs / lag)


def update_region_estimates(region):
    state = region_states[region]

    if (
        len(state["pos_filtered"]) >= hr_window
        and state["pos_sample_count"] - state["pos_last_update"] >= update_interval
    ):
        hr, conf, status, previous, pending, pending_count = estimate_hr_from_signal(
            state["pos_filtered"], state["pos_history"], state["pos_previous_hr"],
            state["pos_pending_hr"], state["pos_pending_count"]
        )
        state["pos_hr"], state["pos_raw_hr"] = hr, hr
        state["pos_confidence"], state["pos_status"] = conf, status
        state["pos_previous_hr"] = previous
        state["pos_pending_hr"], state["pos_pending_count"] = pending, pending_count
        state["pos_peak_diagnostics"], state["pos_autocorr_bpm"] = signal_diagnostics(state["pos_filtered"])
        state["pos_last_update"] = state["pos_sample_count"]

    if (
        len(state["chrom_filtered"]) >= hr_window
        and state["chrom_sample_count"] - state["chrom_last_update"] >= update_interval
    ):
        hr, conf, status, previous, pending, pending_count = estimate_hr_from_signal(
            state["chrom_filtered"], state["chrom_history"], state["chrom_previous_hr"],
            state["chrom_pending_hr"], state["chrom_pending_count"]
        )
        state["chrom_hr"], state["chrom_raw_hr"] = hr, hr
        state["chrom_confidence"], state["chrom_status"] = conf, status
        state["chrom_previous_hr"] = previous
        state["chrom_pending_hr"], state["chrom_pending_count"] = pending, pending_count
        state["chrom_peak_diagnostics"], state["chrom_autocorr_bpm"] = signal_diagnostics(state["chrom_filtered"])
        state["chrom_last_update"] = state["chrom_sample_count"]


def get_algorithm_combined(key):
    values = []
    confidences = []
    for region in REGIONS:
        state = region_states[region]
        hr = state[f"{key}_hr"]
        conf = state[f"{key}_confidence"]
        status = state[f"{key}_status"]
        if hr is not None and conf >= CONFIDENCE_THRESHOLD and status == "OK":
            values.append(hr)
            confidences.append(conf)
    if len(values) < 2 or (max(values) - min(values)) > 12.0:
        return None, None
    return float(np.median(values)), float(np.median(confidences))


# ============================================================
# TKINTER USER INTERFACE
# ============================================================

root = tk.Tk()
root.title("FaceVitals - POS + CHROM Heart Rate Monitor")
root.geometry("1220x820")
root.minsize(1100, 760)
root.configure(bg="#101820")

title = tk.Label(
    root, text="FaceVitals", font=("Arial", 27, "bold"),
    fg="white", bg="#101820"
)
title.pack(pady=(8, 0))

subtitle = tk.Label(
    root, text="Contactless Heart Rate Monitoring | POS + CHROM",
    font=("Arial", 11), fg="#AAB7C4", bg="#101820"
)
subtitle.pack()

main_frame = tk.Frame(root, bg="#101820")
main_frame.pack(padx=12, pady=10, fill="both", expand=True)

camera_frame = tk.LabelFrame(
    main_frame, text="Camera / Video Preview",
    font=("Arial", 12, "bold"), fg="white", bg="#1B2836",
    width=700, height=470
)
camera_frame.pack(side="left", padx=8, fill="both", expand=True)
camera_frame.pack_propagate(False)

camera_label = tk.Label(
    camera_frame, text="Camera is stopped",
    font=("Arial", 15), fg="#AAB7C4", bg="#1B2836"
)
camera_label.pack(expand=True)

results_frame = tk.LabelFrame(
    main_frame, text="Measurement Results",
    font=("Arial", 12, "bold"), fg="white", bg="#1B2836",
    width=440, height=470
)
results_frame.pack(side="right", padx=8, fill="y")
results_frame.pack_propagate(False)

def make_label(parent, text, font, fg="white", pady=2):
    label = tk.Label(parent, text=text, font=font, fg=fg, bg="#1B2836")
    label.pack(pady=pady)
    return label


make_label(results_frame, "POS HEART RATE", ("Arial", 10, "bold"), "#AAB7C4", 8)
pos_total_label = make_label(results_frame, "-- BPM", ("Arial", 20, "bold"), "#53B6FF")
make_label(results_frame, "CHROM HEART RATE", ("Arial", 10, "bold"), "#AAB7C4", 5)
chrom_total_label = make_label(results_frame, "-- BPM", ("Arial", 20, "bold"), "#B69CFF")
make_label(results_frame, "FINAL COMBINED HEART RATE", ("Arial", 11, "bold"), "#AAB7C4", 8)
final_hr_label = make_label(results_frame, "-- BPM", ("Arial", 29, "bold"), "#00D084", 2)
final_confidence_label = make_label(results_frame, "Confidence: --", ("Arial", 10))
final_status_label = make_label(
    results_frame, "Status: Not measuring", ("Arial", 10, "bold"), "#FFC857", 4
)

make_label(results_frame, "REGIONAL RESULTS", ("Arial", 11, "bold"), "#00D084", 10)

region_labels = {}
for region in REGIONS:
    make_label(results_frame, region, ("Arial", 10, "bold"), "white", 4)
    region_labels[region] = make_label(
        results_frame, "POS: -- | CHROM: --", ("Arial", 9), "#E6EDF3", 1
    )
    region_labels[region + "_confidence"] = make_label(
        results_frame, "POS conf: -- | CHROM conf: --", ("Arial", 8), "#AAB7C4", 1
    )
    region_labels[region + "_status"] = make_label(
        results_frame, "Warming up", ("Arial", 8), "#AAB7C4", 1
    )

source_label = tk.Label(
    root, text="Source: None", font=("Arial", 9),
    fg="#AAB7C4", bg="#101820"
)
source_label.pack(pady=(0, 3))


# ============================================================
# CSV EXPORT
# ============================================================

def save_results():
    if not record_rows:
        messagebox.showinfo("No Data", "There are no measurement records to save yet.")
        return

    filename = "facevitals_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    output_path = os.path.join(BASE_DIR, filename)
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(record_rows[0].keys()))
            writer.writeheader()
            writer.writerows(record_rows)
        messagebox.showinfo("Saved", f"Measurement data saved:\n{output_path}")
    except OSError as error:
        messagebox.showerror("Save Error", str(error))


# ============================================================
# FRAME PROCESSING SHARED BY WEBCAM AND UPLOADED VIDEO
# ============================================================

def process_frame(frame, elapsed_seconds, actual_fps):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    global last_mp_timestamp
    timestamp_ms = max(int(elapsed_seconds * 1000), last_mp_timestamp + 1)
    last_mp_timestamp = timestamp_ms

    detection = landmarker.detect_for_video(mp_image, timestamp_ms)
    face_detected = bool(detection.face_landmarks)
    current_rgb = {}

    if face_detected:
        landmarks = detection.face_landmarks[0]
        height, width = frame.shape[:2]
        xs = [int(p.x * width) for p in landmarks]
        ys = [int(p.y * height) for p in landmarks]
        bbox = (
            max(0, min(xs)), max(0, min(ys)),
            min(width - 1, max(xs)), min(height - 1, max(ys))
        )

        for region in REGIONS:
            mean_rgb, coordinates, outline = extract_roi_rgb(
                rgb_frame, landmarks, bbox, region
            )

            if coordinates is not None:
                x1, y1, x2, y2 = coordinates
                if outline is not None:
                    cv2.polylines(
                        frame, [np.asarray(outline, dtype=np.int32)],
                        isClosed=True, color=(255, 255, 0), thickness=2
                    )
                cv2.putText(
                    frame, region, (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1
                )

            if mean_rgb is not None:
                current_rgb[region] = mean_rgb
                process_pos(region, mean_rgb)
                process_chrom(region, mean_rgb)
                update_region_estimates(region)
            else:
                region_states[region] = create_region_state()

        cv2.putText(
            frame, "FACE DETECTED", (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    else:
        reset_all_regions()
        cv2.putText(
            frame, "NO FACE DETECTED", (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

    pos_hr, pos_conf = get_algorithm_combined("pos")
    chrom_hr, chrom_conf = get_algorithm_combined("chrom")

    if pos_hr is not None:
        pos_total_label.config(text=f"{pos_hr:.1f} BPM")
    else:
        pos_total_label.config(text="-- BPM")

    if chrom_hr is not None:
        chrom_total_label.config(text=f"{chrom_hr:.1f} BPM")
    else:
        chrom_total_label.config(text="-- BPM")

    final_hr = None
    final_conf = None
    final_status = "Collecting reliable POS and CHROM estimates"

    if not face_detected:
        final_status = "No face detected"
    elif pos_hr is not None and chrom_hr is not None:
        difference = abs(pos_hr - chrom_hr)
        if difference <= MAX_ALGORITHM_DIFFERENCE:
            final_hr = (pos_hr + chrom_hr) / 2.0
            final_conf = float(np.median([pos_conf, chrom_conf]))
            final_status = "Reliable estimate"
        else:
            final_status = f"POS / CHROM disagree ({difference:.1f} BPM)"
    elif pos_hr is not None or chrom_hr is not None:
        final_status = "Waiting for both algorithms"

    if final_hr is not None:
        final_hr_label.config(text=f"{final_hr:.1f} BPM")
        final_confidence_label.config(text=f"Confidence: {final_conf:.3f}")
        final_status_label.config(text=f"Status: {final_status}", fg="#00D084")
    else:
        final_hr_label.config(text="-- BPM")
        final_confidence_label.config(text="Confidence: --")
        final_status_label.config(text=f"Status: {final_status}", fg="#FFC857")

    for region in REGIONS:
        state = region_states[region]
        p = state["pos_hr"]
        c = state["chrom_hr"]
        p_text = f"{p:.1f}" if p is not None else "--"
        c_text = f"{c:.1f}" if c is not None else "--"
        region_labels[region].config(text=f"POS: {p_text} BPM | CHROM: {c_text} BPM")
        region_labels[region + "_confidence"].config(
            text=(
                f"POS conf: {state['pos_confidence']:.2f} | "
                f"CHROM conf: {state['chrom_confidence']:.2f}"
            )
        )
        region_labels[region + "_status"].config(
            text=f"POS: {state['pos_status']} | CHROM: {state['chrom_status']}"
        )

    row = {
        "Elapsed_Time_s": float(elapsed_seconds),
        "Source_FPS": float(actual_fps),
        "Face_Detected": bool(face_detected),
        "POS_Combined_BPM": pos_hr,
        "POS_Confidence": pos_conf,
        "CHROM_Combined_BPM": chrom_hr,
        "CHROM_Confidence": chrom_conf,
        "Final_Combined_BPM": final_hr,
        "Final_Confidence": final_conf,
        "Final_Status": final_status,
    }

    for region in REGIONS:
        state = region_states[region]
        row[f"{region}_POS_Top5Peaks_BPM_Power"] = state["pos_peak_diagnostics"]
        row[f"{region}_POS_Autocorr_BPM"] = state["pos_autocorr_bpm"]
        row[f"{region}_CHROM_Top5Peaks_BPM_Power"] = state["chrom_peak_diagnostics"]
        row[f"{region}_CHROM_Autocorr_BPM"] = state["chrom_autocorr_bpm"]

    for region in REGIONS:
        state = region_states[region]
        rgb = current_rgb.get(region)
        row[f"{region}_R"] = float(rgb[0]) if rgb is not None else np.nan
        row[f"{region}_G"] = float(rgb[1]) if rgb is not None else np.nan
        row[f"{region}_B"] = float(rgb[2]) if rgb is not None else np.nan
        row[f"{region}_POS_BPM"] = state["pos_hr"]
        row[f"{region}_POS_Confidence"] = state["pos_confidence"]
        row[f"{region}_POS_Status"] = state["pos_status"]
        row[f"{region}_CHROM_BPM"] = state["chrom_hr"]
        row[f"{region}_CHROM_Confidence"] = state["chrom_confidence"]
        row[f"{region}_CHROM_Status"] = state["chrom_status"]
    record_rows.append(row)

    return frame


# ============================================================
# SOURCE CONTROL / FRAME LOOP
# ============================================================

def show_frame(frame):
    frame = cv2.resize(frame, (660, 390))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(Image.fromarray(frame))
    camera_label.config(image=photo, text="")
    camera_label.image = photo


def source_update():
    global source_running, capture, video_frame_index

    if not source_running or capture is None:
        return

    ret, frame = capture.read()
    if not ret:
        stop_source()
        if source_mode == "video":
            final_status_label.config(text="Status: Video analysis complete", fg="#00D084")
            messagebox.showinfo("Complete", "Video analysis has finished. You can save the CSV results.")
        return

    if source_mode == "camera":
        now = time.perf_counter()
        camera_times.append(now)
        elapsed = now - session_start
        if len(camera_times) >= 2:
            duration = camera_times[-1] - camera_times[0]
            actual_fps = (len(camera_times) - 1) / duration if duration > 0 else 0.0
        else:
            actual_fps = 0.0
        delay = 15
    else:
        elapsed = video_frame_index / source_fps
        actual_fps = source_fps
        video_frame_index += 1
        delay = max(1, int(1000.0 / source_fps))

    processed = process_frame(frame, elapsed, actual_fps)
    show_frame(processed)
    root.after(delay, source_update)


def start_camera():
    global capture, source_mode, source_running, session_start

    if source_running:
        stop_source()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        messagebox.showerror("Camera Error", "Could not access the webcam.")
        return

    configure_sampling(FS_DEFAULT)
    reset_all_regions()
    record_rows.clear()
    camera_times.clear()
    capture = cap
    source_mode = "camera"
    source_running = True
    session_start = time.perf_counter()
    source_label.config(text="Source: Live webcam")
    final_status_label.config(text="Status: Warming up", fg="#FFC857")
    source_update()


def upload_video():
    global capture, source_mode, source_running, source_fps
    global video_frame_index, video_path

    path = filedialog.askopenfilename(
        title="Select a video for FaceVitals analysis",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
            ("All files", "*.*"),
        ],
    )
    if not path:
        return

    if source_running:
        stop_source()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        messagebox.showerror("Video Error", "Could not open the selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 10 or fps > 60:
        cap.release()
        messagebox.showerror(
            "Unsupported Video FPS",
            f"Detected video FPS: {fps:.2f}\nPlease use a video between 10 and 60 FPS."
        )
        return

    configure_sampling(fps)
    reset_all_regions()
    record_rows.clear()
    camera_times.clear()
    capture = cap
    source_mode = "video"
    source_running = True
    source_fps = float(fps)
    video_frame_index = 0
    video_path = path
    source_label.config(text=f"Source: {os.path.basename(path)} | {fps:.2f} FPS")
    final_status_label.config(text="Status: Analyzing uploaded video", fg="#FFC857")
    source_update()


def stop_source():
    global capture, source_running
    source_running = False
    if capture is not None:
        capture.release()
        capture = None
    camera_label.config(image="", text="Source is stopped")
    camera_label.image = None


# ============================================================
# BUTTONS / SHUTDOWN
# ============================================================

controls = tk.Frame(root, bg="#101820")
controls.pack(pady=8)

def make_button(text, command, color):
    tk.Button(
        controls, text=text, command=command,
        font=("Arial", 11, "bold"), bg=color, fg="white",
        width=15, relief="flat", padx=4, pady=7
    ).pack(side="left", padx=6)


make_button("START WEBCAM", start_camera, "#00A86B")
make_button("UPLOAD VIDEO", upload_video, "#2874A6")
make_button("STOP", stop_source, "#C0392B")
make_button("SAVE CSV", save_results, "#6C5CE7")


def close_application():
    stop_source()
    try:
        landmarker.close()
    except Exception:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_application)
root.mainloop()
