
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from datetime import datetime
from collections import deque
from PIL import Image, ImageTk
from scipy.signal import butter, sosfilt, sosfilt_zi


# =====================================================
# FACEVITALS - INTEGRATED LIVE POS APPLICATION
# Camera + Face Detection + RGB + POS + HR + Confidence
# =====================================================

FS = 30.0
POS_WINDOW_SECONDS = 1.6
HR_WINDOW_SECONDS = 10.0
FFT_SIZE = 2048

POS_WINDOW = int(FS * POS_WINDOW_SECONDS)
HR_WINDOW = int(FS * HR_WINDOW_SECONDS)

LOW_HR = 0.7
HIGH_HR = 3.5
CONFIDENCE_THRESHOLD = 0.30

FILTER_ORDER = 4
UPDATE_INTERVAL = 30

ROIS = {
    "Forehead": (0.30, 0.08, 0.70, 0.27),
    "Left Cheek": (0.12, 0.40, 0.42, 0.62),
    "Right Cheek": (0.58, 0.40, 0.88, 0.62)
}

REGIONS = list(ROIS.keys())

# =====================================================
# APPLICATION STATE
# =====================================================

camera = None
camera_running = False
start_time = 0.0
last_mp_timestamp = -1

record_rows = []

camera_times = deque(maxlen=60)

filter_sos = butter(
    FILTER_ORDER,
    [LOW_HR, HIGH_HR],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def create_region_state():

    return {
        "rgb": deque(maxlen=POS_WINDOW),

        "overlap_sum": np.zeros(POS_WINDOW),
        "overlap_count": np.zeros(POS_WINDOW),

        "filter_zi": None,

        "filtered": deque(maxlen=HR_WINDOW),

        "hr_history": deque(maxlen=5),

        "sample_count": 0,
        "last_hr_update": 0,

        "raw_hr": None,
        "hr": None,
        "confidence": 0.0,
        "status": "Warming up",

        "previous_hr": None
    }


region_states = {
    region: create_region_state()
    for region in REGIONS
}


def reset_region(region):

    region_states[region] = create_region_state()


def reset_all_regions():

    for region in REGIONS:
        reset_region(region)


# =====================================================
# MEDIAPIPE MODEL
# =====================================================

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "face_landmarker.task was not found in the project folder."
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# =====================================================
# SKIN MASK AND RGB EXTRACTION
# =====================================================

def create_skin_mask(rgb_image):

    hsv = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2HSV
    )

    lower1 = np.array([0, 20, 40])
    upper1 = np.array([25, 200, 255])

    lower2 = np.array([160, 20, 40])
    upper2 = np.array([180, 200, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    return cv2.bitwise_or(mask1, mask2)


def extract_roi_rgb(rgb_frame, bbox, roi):

    x_min, y_min, x_max, y_max = bbox

    face_width = x_max - x_min
    face_height = y_max - y_min

    rx1, ry1, rx2, ry2 = roi

    x1 = max(0, int(x_min + rx1 * face_width))
    y1 = max(0, int(y_min + ry1 * face_height))

    x2 = min(
        rgb_frame.shape[1],
        int(x_min + rx2 * face_width)
    )

    y2 = min(
        rgb_frame.shape[0],
        int(y_min + ry2 * face_height)
    )

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi_image = rgb_frame[y1:y2, x1:x2]

    if roi_image.size == 0:
        return None, None

    skin_mask = create_skin_mask(roi_image)
    skin_pixels = roi_image[skin_mask > 0]

    total_pixels = roi_image.shape[0] * roi_image.shape[1]

    skin_ratio = len(skin_pixels) / total_pixels

    if skin_ratio < 0.10 or len(skin_pixels) == 0:
        return None, (x1, y1, x2, y2)

    mean_rgb = np.mean(skin_pixels, axis=0)

    return mean_rgb, (x1, y1, x2, y2)


# =====================================================
# POS ALGORITHM - LIVE OVERLAP ADDITION
# =====================================================

def process_pos(region, rgb_values):

    state = region_states[region]

    state["rgb"].append(rgb_values)

    if len(state["rgb"]) < POS_WINDOW:
        return None

    rgb_window = np.asarray(
        state["rgb"],
        dtype=float
    )

    # Normalize each color channel by its window mean
    channel_means = np.mean(rgb_window, axis=0)

    if np.any(channel_means < 1e-8):
        reset_region(region)
        return None

    normalized = rgb_window / channel_means

    Rn = normalized[:, 0]
    Gn = normalized[:, 1]
    Bn = normalized[:, 2]

    # POS projection
    S1 = Gn - Bn
    S2 = Gn + Bn - 2.0 * Rn

    std_S2 = np.std(S2)

    if std_S2 < 1e-8:
        reset_region(region)
        return None

    alpha = np.std(S1) / std_S2

    h = S1 + alpha * S2
    h = h - np.mean(h)

    # Overlap-add
    state["overlap_sum"] += h
    state["overlap_count"] += 1

    count = state["overlap_count"][0]

    if count == 0:
        return None

    pos_sample = (
        state["overlap_sum"][0] / count
    )

    # Move overlap buffers forward by one sample
    state["overlap_sum"][:-1] = state["overlap_sum"][1:]
    state["overlap_sum"][-1] = 0.0

    state["overlap_count"][:-1] = state["overlap_count"][1:]
    state["overlap_count"][-1] = 0.0

    # Causal filtering for live processing
    if state["filter_zi"] is None:

        state["filter_zi"] = (
            sosfilt_zi(filter_sos) * pos_sample
        )

    filtered_sample, state["filter_zi"] = sosfilt(
        filter_sos,
        [pos_sample],
        zi=state["filter_zi"]
    )

    filtered_value = float(filtered_sample[0])

    state["filtered"].append(filtered_value)

    state["sample_count"] += 1

    return filtered_value


# =====================================================
# HEART RATE AND CONFIDENCE ESTIMATION
# =====================================================

def estimate_hr(region):

    state = region_states[region]

    if len(state["filtered"]) < HR_WINDOW:
        return

    signal = np.asarray(
        state["filtered"],
        dtype=float
    )

    # Remove DC component
    signal = signal - np.mean(signal)

    # Hann window
    window = np.hanning(HR_WINDOW)

    windowed_signal = signal * window

    # Real FFT
    spectrum = np.fft.rfft(
        windowed_signal,
        n=FFT_SIZE
    )

    power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(
        FFT_SIZE,
        d=1.0 / FS
    )

    valid = (
        (frequencies >= LOW_HR) &
        (frequencies <= HIGH_HR)
    )

    if not np.any(valid):
        return

    valid_indices = np.where(valid)[0]

    peak_index = valid_indices[
        np.argmax(power[valid])
    ]

    peak_frequency = frequencies[peak_index]

    raw_hr = peak_frequency * 60.0

    total_power = np.sum(power[valid])

    if total_power <= 1e-12:
        return

    # Power near fundamental frequency
    fundamental_mask = (
        np.abs(frequencies - peak_frequency) <= 0.1
    )

    # Power near second harmonic
    second_harmonic = 2.0 * peak_frequency

    harmonic_mask = (
        np.abs(frequencies - second_harmonic) <= 0.1
    ) & valid

    fundamental_power = np.sum(
        power[fundamental_mask & valid]
    )

    harmonic_power = np.sum(
        power[harmonic_mask]
    )

    confidence = (
        fundamental_power + harmonic_power
    ) / total_power

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    state["raw_hr"] = float(raw_hr)
    state["confidence"] = confidence

    # Median of the latest five estimates
    state["hr_history"].append(raw_hr)

    smoothed_hr = float(
        np.median(state["hr_history"])
    )

    previous_hr = state["previous_hr"]

    if confidence < CONFIDENCE_THRESHOLD:

        status = "Low confidence"

    elif (
        previous_hr is not None and
        abs(smoothed_hr - previous_hr) > 20.0
    ):

        status = "Jump warning"

    else:

        status = "OK"

    state["hr"] = smoothed_hr
    state["status"] = status
    state["previous_hr"] = smoothed_hr


def update_hr_if_ready(region):

    state = region_states[region]

    if len(state["filtered"]) < HR_WINDOW:
        state["status"] = "Warming up"
        return

    if (
        state["sample_count"] -
        state["last_hr_update"]
    ) >= UPDATE_INTERVAL:

        estimate_hr(region)

        state["last_hr_update"] = state["sample_count"]


# =====================================================
# APPLICATION INTERFACE
# =====================================================

root = tk.Tk()

root.title("FaceVitals - Live Heart Rate Monitor")
root.geometry("1100x750")
root.configure(bg="#101820")
root.resizable(False, False)

title = tk.Label(
    root,
    text="FaceVitals",
    font=("Arial", 28, "bold"),
    fg="white",
    bg="#101820"
)
title.pack(pady=(10, 0))

subtitle = tk.Label(
    root,
    text="Contactless Heart Rate Monitoring",
    font=("Arial", 12),
    fg="#AAB7C4",
    bg="#101820"
)
subtitle.pack()

main_frame = tk.Frame(
    root,
    bg="#101820"
)

main_frame.pack(
    pady=15,
    padx=20,
    fill="both",
    expand=True
)


# =====================================================
# CAMERA PREVIEW
# =====================================================

camera_frame = tk.LabelFrame(
    main_frame,
    text="Live Camera and Facial ROIs",
    font=("Arial", 13, "bold"),
    fg="white",
    bg="#1B2836",
    width=650,
    height=430
)

camera_frame.pack(
    side="left",
    padx=10,
    fill="both",
    expand=True
)

camera_frame.pack_propagate(False)

camera_label = tk.Label(
    camera_frame,
    text="Camera is stopped",
    font=("Arial", 16),
    fg="#AAB7C4",
    bg="#1B2836"
)

camera_label.pack(expand=True)


# =====================================================
# RESULTS PANEL
# =====================================================

results_frame = tk.LabelFrame(
    main_frame,
    text="Live Measurement Results",
    font=("Arial", 13, "bold"),
    fg="white",
    bg="#1B2836",
    width=340,
    height=430
)

results_frame.pack(
    side="right",
    padx=10,
    fill="both"
)

results_frame.pack_propagate(False)

tk.Label(
    results_frame,
    text="COMBINED HEART RATE",
    font=("Arial", 11, "bold"),
    fg="#AAB7C4",
    bg="#1B2836"
).pack(pady=(12, 0))

hr_value = tk.Label(
    results_frame,
    text="-- BPM",
    font=("Arial", 30, "bold"),
    fg="#00D084",
    bg="#1B2836"
)

hr_value.pack(pady=3)

confidence_label = tk.Label(
    results_frame,
    text="Confidence: --",
    font=("Arial", 11),
    fg="white",
    bg="#1B2836"
)

confidence_label.pack(pady=3)

status_label = tk.Label(
    results_frame,
    text="Status: Not measuring",
    font=("Arial", 11, "bold"),
    fg="#FFC857",
    bg="#1B2836"
)

status_label.pack(pady=5)

tk.Label(
    results_frame,
    text="REGIONAL RESULTS",
    font=("Arial", 12, "bold"),
    fg="#00D084",
    bg="#1B2836"
).pack(pady=(12, 5))

hr_labels = {}
rgb_labels = {}
confidence_labels = {}

for region in REGIONS:

    tk.Label(
        results_frame,
        text=region,
        font=("Arial", 11, "bold"),
        fg="white",
        bg="#1B2836"
    ).pack(pady=(5, 0))

    hr_labels[region] = tk.Label(
        results_frame,
        text="HR: -- BPM",
        font=("Arial", 10),
        fg="#00D084",
        bg="#1B2836"
    )

    hr_labels[region].pack()

    confidence_labels[region] = tk.Label(
        results_frame,
        text="Confidence: -- | Warming up",
        font=("Arial", 9),
        fg="#AAB7C4",
        bg="#1B2836"
    )

    confidence_labels[region].pack()

    rgb_labels[region] = tk.Label(
        results_frame,
        text="RGB: --",
        font=("Arial", 9),
        fg="white",
        bg="#1B2836"
    )

    rgb_labels[region].pack()


# =====================================================
# CSV RECORDING
# =====================================================

def save_results():

    if not record_rows:

        messagebox.showinfo(
            "No Data",
            "There are no measurement records to save yet."
        )
        return

    filename = (
        "facevitals_live_" +
        datetime.now().strftime("%Y%m%d_%H%M%S") +
        ".csv"
    )

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename
    )

    try:

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(record_rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(record_rows)

        messagebox.showinfo(
            "Saved",
            f"Measurement data saved:\n{output_path}"
        )

    except OSError as error:

        messagebox.showerror(
            "Save Error",
            str(error)
        )


# =====================================================
# CAMERA UPDATE AND LIVE PROCESSING
# =====================================================

def update_camera():

    global camera_running, last_mp_timestamp

    if not camera_running or camera is None:
        return

    ret, frame = camera.read()

    if not ret:

        stop_measurement()

        status_label.config(
            text="Status: Camera error",
            fg="red"
        )
        return

    now = time.perf_counter()

    camera_times.append(now)

    # Actual camera FPS estimate
    if len(camera_times) >= 2:

        elapsed = camera_times[-1] - camera_times[0]

        actual_fps = (
            (len(camera_times) - 1) / elapsed
            if elapsed > 0 else 0
        )

    else:

        actual_fps = 0

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (now - start_time) * 1000
    )

    timestamp_ms = max(
        timestamp_ms,
        last_mp_timestamp + 1
    )

    last_mp_timestamp = timestamp_ms

    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    face_detected = bool(results.face_landmarks)

    current_rgb = {}

    if face_detected:

        landmarks = results.face_landmarks[0]

        height, width, _ = frame.shape

        xs = [int(p.x * width) for p in landmarks]
        ys = [int(p.y * height) for p in landmarks]

        bbox = (
            max(0, min(xs)),
            max(0, min(ys)),
            min(width - 1, max(xs)),
            min(height - 1, max(ys))
        )

        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2
        )

        for region, roi in ROIS.items():

            mean_rgb, coordinates = extract_roi_rgb(
                rgb_frame,
                bbox,
                roi
            )

            if coordinates is not None:

                x1, y1, x2, y2 = coordinates

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    region,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

            if mean_rgb is not None:

                current_rgb[region] = mean_rgb

                r, g, b = mean_rgb

                rgb_labels[region].config(
                    text=(
                        f"RGB: {r:.1f}, "
                        f"{g:.1f}, {b:.1f}"
                    )
                )

                process_pos(
                    region,
                    mean_rgb
                )

                update_hr_if_ready(region)

            else:

                reset_region(region)

                rgb_labels[region].config(
                    text="RGB: Invalid skin region"
                )

        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        # Reset signals after loss of face.
        reset_all_regions()

        for region in REGIONS:

            rgb_labels[region].config(
                text="RGB: --"
            )

            hr_labels[region].config(
                text="HR: -- BPM"
            )

            confidence_labels[region].config(
                text="Confidence: -- | No face"
            )

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # =================================================
    # UPDATE REGIONAL RESULTS
    # =================================================

    valid_hrs = []
    valid_confidences = []

    for region in REGIONS:

        state = region_states[region]

        if state["hr"] is not None:

            hr_labels[region].config(
                text=f"HR: {state['hr']:.1f} BPM"
            )

            confidence_labels[region].config(
                text=(
                    f"Confidence: {state['confidence']:.3f} "
                    f"| {state['status']}"
                )
            )

            if state["status"] == "OK":

                valid_hrs.append(state["hr"])
                valid_confidences.append(
                    state["confidence"]
                )

        else:

            if face_detected and region in current_rgb:

                samples = state["sample_count"]

                hr_labels[region].config(
                    text="HR: -- BPM"
                )

                confidence_labels[region].config(
                    text=(
                        f"Collecting: {samples}/{HR_WINDOW} "
                        f"POS samples"
                    )
                )

            else:

                hr_labels[region].config(
                    text="HR: -- BPM"
                )

    # =================================================
    # COMBINED HEART RATE
    # =================================================

    if not face_detected:

        hr_value.config(text="-- BPM")

        confidence_label.config(
            text="Confidence: --"
        )

        status_label.config(
            text="Status: No face",
            fg="#FFC857"
        )

    elif actual_fps > 0 and actual_fps < 20:

        status_label.config(
            text=f"Status: Low FPS ({actual_fps:.1f})",
            fg="#FFC857"
        )

    elif valid_hrs:

        combined_hr = float(
            np.median(valid_hrs)
        )

        combined_confidence = float(
            np.median(valid_confidences)
        )

        hr_value.config(
            text=f"{combined_hr:.1f} BPM"
        )

        confidence_label.config(
            text=f"Confidence: {combined_confidence:.3f}"
        )

        status_label.config(
            text="Status: Measuring - OK",
            fg="#00D084"
        )

    else:

        all_ready = all(
            len(region_states[r]["filtered"]) >= HR_WINDOW
            for r in REGIONS
        )

        hr_value.config(text="-- BPM")

        confidence_label.config(
            text="Confidence: --"
        )

        if all_ready:

            status_label.config(
                text="Status: Low confidence",
                fg="#FFC857"
            )

        else:

            status_label.config(
                text="Status: Measuring / Warming up",
                fg="#FFC857"
            )

    # =================================================
    # RECORD DATA
    # =================================================

    row = {
        "Time": now - start_time,
        "Camera_FPS": actual_fps,
        "Face_Detected": face_detected
    }

    for region in REGIONS:

        state = region_states[region]

        rgb = current_rgb.get(region)

        if rgb is not None:

            row[f"{region}_R"] = float(rgb[0])
            row[f"{region}_G"] = float(rgb[1])
            row[f"{region}_B"] = float(rgb[2])

        else:

            row[f"{region}_R"] = np.nan
            row[f"{region}_G"] = np.nan
            row[f"{region}_B"] = np.nan

        row[f"{region}_HR"] = state["hr"]
        row[f"{region}_Confidence"] = state["confidence"]
        row[f"{region}_Status"] = state["status"]

    record_rows.append(row)

    # =================================================
    # DISPLAY CAMERA
    # =================================================

    frame = cv2.resize(
        frame,
        (620, 370)
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame)

    photo = ImageTk.PhotoImage(image)

    camera_label.config(
        image=photo,
        text=""
    )

    camera_label.image = photo

    root.after(15, update_camera)


# =====================================================
# START AND STOP
# =====================================================

def start_measurement():

    global camera
    global camera_running
    global start_time
    global last_mp_timestamp

    if camera_running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        camera.release()
        camera = None

        messagebox.showerror(
            "Camera Error",
            "Could not access webcam."
        )
        return

    camera.set(
        cv2.CAP_PROP_FPS,
        FS
    )

    reset_all_regions()

    record_rows.clear()
    camera_times.clear()

    start_time = time.perf_counter()
    last_mp_timestamp = -1

    camera_running = True

    status_label.config(
        text="Status: Measuring / Warming up",
        fg="#FFC857"
    )

    update_camera()


def stop_measurement():

    global camera
    global camera_running

    camera_running = False

    if camera is not None:

        camera.release()
        camera = None

    camera_label.config(
        image="",
        text="Camera is stopped"
    )

    camera_label.image = None

    status_label.config(
        text="Status: Stopped",
        fg="#FFC857"
    )


# =====================================================
# BUTTONS
# =====================================================

control_frame = tk.Frame(
    root,
    bg="#101820"
)

control_frame.pack(pady=10)

tk.Button(
    control_frame,
    text="START",
    command=start_measurement,
    font=("Arial", 12, "bold"),
    bg="#00A86B",
    fg="white",
    width=13,
    relief="flat"
).pack(side="left", padx=10)

tk.Button(
    control_frame,
    text="STOP",
    command=stop_measurement,
    font=("Arial", 12, "bold"),
    bg="#C0392B",
    fg="white",
    width=13,
    relief="flat"
).pack(side="left", padx=10)

tk.Button(
    control_frame,
    text="SAVE CSV",
    command=save_results,
    font=("Arial", 12, "bold"),
    bg="#2874A6",
    fg="white",
    width=13,
    relief="flat"
).pack(side="left", padx=10)


# =====================================================
# CLOSE APPLICATION
# =====================================================

def close_application():

    stop_measurement()

    landmarker.close()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_application
)

root.mainloop()




