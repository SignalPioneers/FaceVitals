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
MIN_SKIN_RATIO = 0.10
MAX_SAMPLE_GAP_SECONDS = 0.15
MIN_VALID_REGIONS = 2
MAX_REGION_SPREAD = 12.0

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
after_id = None
last_reliable_result = None
latest_elapsed_seconds = 0.0
combined_history = deque(maxlen=5)
last_combined_history_second = -1

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
        "chrom_overlap_sum": np.zeros(pos_window, dtype=float),
        "chrom_overlap_count": np.zeros(pos_window, dtype=float),
        "chrom_filtered": deque(maxlen=hr_window),
        "rgb_time_buffer": deque(maxlen=2),
        "next_sample_time": None,
        "last_rgb": None,
        "last_rgb_time": None,
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
        "pos_last_good_time": None,
        "chrom_last_good_time": None,
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
            return None, None, None, 0.0

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
        return None, None, None, 0.0

    if x2 <= x1 or y2 <= y1:
        return None, None, None, 0.0

    roi_image = rgb_frame[y1:y2, x1:x2]
    if roi_image.size == 0:
        return None, (x1, y1, x2, y2), None, 0.0

    skin_mask = create_skin_mask(roi_image)
    selected = skin_mask > 0
    skin_pixels = roi_image[selected]
    total_pixels = int(roi_image.shape[0] * roi_image.shape[1])
    skin_ratio = (len(skin_pixels) / total_pixels) if total_pixels else 0.0
    minimum_pixels = max(20, int(total_pixels * MIN_SKIN_RATIO))

    outline = np.array(
        [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
        dtype=np.int32
    )
    if len(skin_pixels) < minimum_pixels:
        return None, (x1, y1, x2, y2), outline, skin_ratio

    mean_rgb = np.mean(skin_pixels, axis=0)
    return mean_rgb, (x1, y1, x2, y2), outline, skin_ratio

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
    X and Y are bandpass-filtered, then every window is overlap-added
    to match the offline CHROM implementation.
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

    # Match the offline CHROM implementation with overlap-add.
    state["chrom_overlap_sum"] += chrom_signal
    state["chrom_overlap_count"] += 1.0
    count = state["chrom_overlap_count"][0]
    if count <= 0:
        return

    chrom_sample = state["chrom_overlap_sum"][0] / count
    state["chrom_overlap_sum"][:-1] = state["chrom_overlap_sum"][1:]
    state["chrom_overlap_sum"][-1] = 0.0
    state["chrom_overlap_count"][:-1] = state["chrom_overlap_count"][1:]
    state["chrom_overlap_count"][-1] = 0.0

    state["chrom_filtered"].append(float(chrom_sample))
    state["chrom_sample_count"] += 1


def process_timestamped_rgb(region, rgb_values, sample_time):
    """Resample RGB onto a uniform time grid before POS and CHROM processing."""
    state = region_states[region]
    buffer = state["rgb_time_buffer"]
    rgb_values = np.asarray(rgb_values, dtype=float)

    if buffer and sample_time <= buffer[-1][0]:
        return

    if buffer and (sample_time - buffer[-1][0]) > MAX_SAMPLE_GAP_SECONDS:
        # Do not stitch pulse signals across a long face/ROI/camera dropout.
        region_states[region] = create_region_state()
        state = region_states[region]
        buffer = state["rgb_time_buffer"]
        buffer.append((float(sample_time), rgb_values))
        state["next_sample_time"] = float(sample_time)
        return

    if not buffer:
        buffer.append((float(sample_time), rgb_values))
        state["next_sample_time"] = float(sample_time)
        return

    buffer.append((float(sample_time), rgb_values))
    t0, rgb0 = buffer[0]
    t1, rgb1 = buffer[1]

    if state["next_sample_time"] is None:
        state["next_sample_time"] = t0

    step = 1.0 / session_fs
    while state["next_sample_time"] <= t1 + 1e-9:
        target_time = max(state["next_sample_time"], t0)
        interval = t1 - t0
        if interval <= 0 or interval > MAX_SAMPLE_GAP_SECONDS:
            break

        fraction = float(np.clip((target_time - t0) / interval, 0.0, 1.0))
        uniform_rgb = rgb0 + fraction * (rgb1 - rgb0)
        process_pos(region, uniform_rgb)
        process_chrom(region, uniform_rgb)
        update_region_estimates(region)
        state["next_sample_time"] += step

    newest = buffer[-1]
    buffer.clear()
    buffer.append(newest)


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

    # Calculate an independent periodicity diagnostic. It is not used as
    # a hard gate because short rPPG windows can produce a different lag.
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
    # Keep autocorrelation as a diagnostic rather than a hard rejection gate.
    # In short/noisy rPPG windows its strongest lag can differ from the FFT
    # fundamental even when the spectral estimate is usable. The FFT
    # confidence and multi-region agreement checks below remain active.

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
        if hr is not None:
            state["pos_hr"], state["pos_raw_hr"] = hr, hr
            state["pos_confidence"], state["pos_status"] = conf, status
            state["pos_last_good_time"] = time.monotonic()
        # If this update fails a quality check, retain the last accepted estimate
        # and its confidence briefly; a single noisy window must not blank the UI.
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
        if hr is not None:
            state["chrom_hr"], state["chrom_raw_hr"] = hr, hr
            state["chrom_confidence"], state["chrom_status"] = conf, status
            state["chrom_last_good_time"] = time.monotonic()
        # If this update fails a quality check, retain the last accepted estimate
        # and its confidence briefly; a single noisy window must not blank the UI.
        state["chrom_previous_hr"] = previous
        state["chrom_pending_hr"], state["chrom_pending_count"] = pending, pending_count
        state["chrom_peak_diagnostics"], state["chrom_autocorr_bpm"] = signal_diagnostics(state["chrom_filtered"])
        state["chrom_last_update"] = state["chrom_sample_count"]


def get_algorithm_combined(key):
    values = []
    confidences = []
    now = time.monotonic()

    for region in REGIONS:
        state = region_states[region]
        hr = state[f"{key}_hr"]
        conf = state[f"{key}_confidence"]
        status = state[f"{key}_status"]
        last_good_time = state[f"{key}_last_good_time"]

        # Uploaded videos are processed frame-by-frame and their final
        # regional estimates remain valid at EOF. For live webcam, reject
        # estimates that have not been refreshed recently.
        if source_mode == "video":
            is_fresh = last_good_time is not None
        else:
            is_fresh = (
                last_good_time is not None
                and (now - last_good_time) <= 5.0
            )

        if (hr is not None and is_fresh
                and conf >= CONFIDENCE_THRESHOLD and status == "OK"):
            values.append(float(hr))
            confidences.append(float(conf))

    if len(values) < MIN_VALID_REGIONS:
        return None, None

    # Require at least two facial regions to agree before publishing an
    # algorithm-level estimate.
    center = float(np.median(values))
    inlier_indices = [
        index for index, value in enumerate(values)
        if abs(value - center) <= MAX_REGION_SPREAD
    ]
    if len(inlier_indices) < MIN_VALID_REGIONS:
        return None, None

    accepted_values = [values[index] for index in inlier_indices]
    accepted_confidences = [confidences[index] for index in inlier_indices]
    if max(accepted_values) - min(accepted_values) > MAX_REGION_SPREAD:
        return None, None
    return (
        float(np.median(accepted_values)),
        float(np.median(accepted_confidences)),
    )


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
    global last_reliable_result, latest_elapsed_seconds, last_combined_history_second
    latest_elapsed_seconds = float(elapsed_seconds)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    global last_mp_timestamp
    timestamp_ms = max(int(elapsed_seconds * 1000), last_mp_timestamp + 1)
    last_mp_timestamp = timestamp_ms

    detection = landmarker.detect_for_video(mp_image, timestamp_ms)
    face_detected = bool(detection.face_landmarks)
    current_rgb = {}
    current_skin_ratios = {}
    current_rgb_changes = {}

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
            mean_rgb, coordinates, outline, skin_ratio = extract_roi_rgb(
                rgb_frame, landmarks, bbox, region
            )
            current_skin_ratios[region] = float(skin_ratio)

            if coordinates is not None:
                x1, y1, x2, y2 = coordinates
                roi_color = (0, 200, 0) if mean_rgb is not None else (0, 0, 255)
                cv2.rectangle(
                    frame, (x1, y1), (x2 - 1, y2 - 1),
                    color=roi_color, thickness=2
                )
                rgb_change = current_rgb_changes.get(region)
                change_text = (
                    f" | RGB Δ {rgb_change:.1f}%"
                    if rgb_change is not None else ""
                )
                cv2.putText(
                    frame, f"{region} skin {skin_ratio:.0%}{change_text}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, roi_color, 1
                )

            if mean_rgb is not None:
                current_rgb[region] = mean_rgb
                previous_rgb = region_states[region]["last_rgb"]
                previous_rgb_time = region_states[region]["last_rgb_time"]
                if (
                    previous_rgb is not None
                    and previous_rgb_time is not None
                    and 0.0 < elapsed_seconds - previous_rgb_time <= MAX_SAMPLE_GAP_SECONDS
                ):
                    relative_change = (
                        np.abs(mean_rgb - previous_rgb) /
                        np.maximum(np.abs(previous_rgb), 1.0)
                    )
                    current_rgb_changes[region] = float(np.mean(relative_change) * 100.0)
                process_timestamped_rgb(region, mean_rgb, elapsed_seconds)
                # Keep these after resampling because a long dropout can reset the region state.
                region_states[region]["last_rgb"] = np.asarray(mean_rgb, dtype=float)
                region_states[region]["last_rgb_time"] = float(elapsed_seconds)
            # Keep the rolling signal state through brief ROI extraction dropouts.
            # Resetting here discarded the 10-second POS/CHROM windows and caused flicker.

        cv2.putText(
            frame, "FACE DETECTED", (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    else:
        # A missed face frame is temporary; do not erase all accumulated signal windows.
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
    final_status = "Collecting POS / CHROM estimates"

    if pos_hr is not None and chrom_hr is not None:
        difference = abs(pos_hr - chrom_hr)
        if difference <= MAX_ALGORITHM_DIFFERENCE:
            final_hr = float((pos_hr + chrom_hr) / 2.0)
            agreement_factor = float(np.clip(1.0 - difference / 40.0, 0.35, 1.0))
            final_conf = float(np.median([pos_conf, chrom_conf]) * agreement_factor)
            final_status = "Estimated BPM — POS and CHROM agree"
        else:
            final_status = (
                f"No combined BPM — POS/CHROM disagree by {difference:.1f} BPM"
            )
    elif not face_detected:
        final_status = "Face temporarily not detected — holding recent valid estimates"
    else:
        final_status = (
            "No combined BPM — need two valid regions from both algorithms"
        )

    if final_hr is not None:
        current_second = int(elapsed_seconds)
        if current_second > last_combined_history_second:
            combined_history.append(float(final_hr))
            last_combined_history_second = current_second
        is_stable = (
            len(combined_history) == combined_history.maxlen
            and (max(combined_history) - min(combined_history)) <= 5.0
        )
        final_hr_label.config(text=f"{final_hr:.1f} BPM")
        final_confidence_label.config(text=f"Estimated confidence: {final_conf:.3f}")
        status_color = "#00D084" if is_stable else "#FFC857"
        stop_message = (
            " — stable for 5 seconds; you can press STOP"
            if is_stable else " — keep still until the value is stable for 5 seconds"
        )
        final_status_label.config(text=f"Status: {final_status}{stop_message}", fg=status_color)
        # Keep the most recent displayed estimate available after the source stops.
        last_reliable_result = (final_hr, final_conf, time.monotonic(), float(elapsed_seconds))
        # Keep measuring until the user presses STOP or closes the window.
    else:
        # A gap breaks the stability streak; require five fresh seconds after recovery.
        combined_history.clear()
        last_combined_history_second = int(elapsed_seconds)
        # Hold the last accepted combined result briefly instead of flickering to blank.
        held = last_reliable_result
        if held is not None and (time.monotonic() - held[2]) <= 5.0:
            final_hr_label.config(text=f"{held[0]:.1f} BPM")
            final_confidence_label.config(text=f"Last accepted confidence: {held[1]:.3f}")
            final_status_label.config(
                text=f"Status: {final_status} — holding last accepted result", fg="#FFC857"
            )
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
            text=(
                "Face temporarily not detected" if not face_detected
                else f"POS: {state['pos_status']} | CHROM: {state['chrom_status']}"
            )
        )

    row = {
        "Elapsed_Time_s": float(elapsed_seconds),
        "Source_FPS": float(actual_fps),
        "Analysis_Sampling_Frequency_Hz": float(session_fs),
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
        row[f"{region}_Skin_Ratio"] = current_skin_ratios.get(region, np.nan)
        row[f"{region}_RGB_Change_Percent"] = current_rgb_changes.get(region, np.nan)
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
    global source_running, capture, video_frame_index, after_id
    after_id = None

    if not source_running or capture is None:
        return

    ret, frame = capture.read()
    if not ret:
        stop_source(manual=False)
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
    if source_running:
        after_id = root.after(delay, source_update)


def start_camera():
    global capture, source_mode, source_running, session_start
    global last_reliable_result, last_combined_history_second

    if source_running:
        stop_source(manual=False)

    cap = cv2.VideoCapture(0)
    # Request HD/30 FPS when supported. Timestamp resampling below keeps
    # the analysis clock correct even if the webcam driver ignores the hints.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, FS_DEFAULT)
    if not cap.isOpened():
        cap.release()
        messagebox.showerror("Camera Error", "Could not access the webcam.")
        return

    configure_sampling(FS_DEFAULT)
    reset_all_regions()
    last_reliable_result = None
    combined_history.clear()
    last_combined_history_second = -1
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
    global last_reliable_result, last_combined_history_second

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
        stop_source(manual=False)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        messagebox.showerror("Video Error", "Could not open the selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 20 or fps > 60:
        cap.release()
        messagebox.showerror(
            "Unsupported Video FPS",
            f"Detected video FPS: {fps:.2f}\nPlease use a video between 20 and 60 FPS."
        )
        return

    configure_sampling(fps)
    reset_all_regions()
    last_reliable_result = None
    combined_history.clear()
    last_combined_history_second = -1
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


def stop_source(manual=True):
    global capture, source_running, after_id
    source_running = False
    if after_id is not None:
        try:
            root.after_cancel(after_id)
        except tk.TclError:
            pass
        after_id = None
    if capture is not None:
        capture.release()
        capture = None
    camera_label.config(image="", text="Source is stopped")
    camera_label.image = None
    if not manual:
        # Automatic EOF, source switching, and window closing should not open dialogs.
        if last_reliable_result is not None and (time.monotonic() - last_reliable_result[2]) <= 5.0:
            final_hr_label.config(text=f"{last_reliable_result[0]:.1f} BPM")
            final_confidence_label.config(text=f"Last accepted confidence: {last_reliable_result[1]:.3f}")
        return

    minimum_seconds = HR_WINDOW_SECONDS + POS_WINDOW_SECONDS
    result = last_reliable_result
    result_is_fresh = result is not None and (time.monotonic() - result[2]) <= 5.0
    enough_data = latest_elapsed_seconds >= minimum_seconds
    stable_at_stop = (
        len(combined_history) == combined_history.maxlen
        and (max(combined_history) - min(combined_history)) <= 5.0
    )

    if result_is_fresh and enough_data:
        bpm, confidence, _accepted_at, accepted_elapsed = result
        final_hr_label.config(text=f"{bpm:.1f} BPM")
        final_confidence_label.config(text=f"Final accepted confidence: {confidence:.3f}")
        stability_text = "stable for 5 seconds" if stable_at_stop else "not yet stable; treat as preliminary"
        final_status_label.config(
            text=f"Status: Stopped — accepted estimate; {stability_text}",
            fg="#00D084" if stable_at_stop else "#FFC857"
        )
        messagebox.showinfo(
            "Measurement finalized",
            f"Final accepted estimate: {bpm:.1f} BPM\n"
            f"Confidence: {confidence:.3f}\n"
            f"Measurement duration: {latest_elapsed_seconds:.1f} seconds\n"
            f"Stability: {stability_text}\n\n"
            "This is an estimate, not a medical diagnosis."
        )
    else:
        final_hr_label.config(text="-- BPM")
        final_confidence_label.config(text="Confidence: --")
        reason = (
            f"Measurement was shorter than the minimum analysis period ({minimum_seconds:.1f}s)."
            if not enough_data else
            "No recent POS + CHROM combined estimate passed the quality checks."
        )
        final_status_label.config(text="Status: Stopped — no reliable final result", fg="#FFC857")
        messagebox.showwarning(
            "No reliable final result",
            reason + "\n\nKeep your face steady, use even lighting, and measure for at least "
            f"{minimum_seconds:.0f} seconds. Stop when POS, CHROM and Combined BPM are present."
        )


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
    stop_source(manual=False)
    try:
        landmarker.close()
    except Exception:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_application)
root.mainloop()
