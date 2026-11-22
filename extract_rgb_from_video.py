
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# ==========================================
# FACEVITALS - OFFLINE VIDEO RGB EXTRACTION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "face_landmarker.task"

INPUT_VIDEO = "C1_low_10.mj2"
OUTPUT_CSV = "C1_10_rgb_retry.csv"
DIAGNOSTIC_IMAGE = "C1_10_ROI_diagnostic.jpg"

# Skin color thresholds (HSV)
LOWER_SKIN_1 = np.array([0, 20, 40])
UPPER_SKIN_1 = np.array([25, 200, 255])

LOWER_SKIN_2 = np.array([160, 20, 40])
UPPER_SKIN_2 = np.array([180, 200, 255])


# ==========================================
# EXTRACT RGB FROM ROI
# ==========================================

def extract_rgb(frame, roi):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_SKIN_1, UPPER_SKIN_1)
    mask2 = cv2.inRange(hsv, LOWER_SKIN_2, UPPER_SKIN_2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = crop.shape[0] * crop.shape[1]

    if skin_pixels < max(5, 0.01 * total_pixels):
        return None

    mean_bgr = cv2.mean(crop, mask=skin_mask)

    return (mean_bgr[2], mean_bgr[1], mean_bgr[0])


# ==========================================
# ROI DIAGNOSTIC
# ==========================================

def diagnose_roi(frame, name, roi):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]

    print(f"\n--- {name} ---")
    print("ROI coordinates:", roi)

    if crop.size == 0:
        print("ERROR: EMPTY ROI")
        return

    print("ROI size (Height, Width):", crop.shape[:2])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_SKIN_1, UPPER_SKIN_1)
    mask2 = cv2.inRange(hsv, LOWER_SKIN_2, UPPER_SKIN_2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = crop.shape[0] * crop.shape[1]

    print("Skin pixels:", skin_pixels)
    print("Total pixels:", total_pixels)

    print("HSV minimum:", hsv.min(axis=(0, 1)))
    print("HSV maximum:", hsv.max(axis=(0, 1)))
    print("HSV mean:", hsv.mean(axis=(0, 1)))

    print("BGR mean:", crop.mean(axis=(0, 1)))


# ==========================================
# MAIN PROCESSING
# ==========================================

def main():
    start_time = time.time()

    if not MODEL_PATH.exists():
        print("ERROR: face_landmarker.task not found.")
        return

    video_path = PROJECT_DIR / INPUT_VIDEO

    if not video_path.exists():
        print(f"ERROR: Video not found: {INPUT_VIDEO}")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if fps <= 0:
        print("ERROR: Invalid video FPS.")
        cap.release()
        return

    print("\n========== FACEVITALS RGB EXTRACTION ==========")
    print(f"Input video: {INPUT_VIDEO}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1
    )

    rows = []

    processed = 0
    detected = 0
    valid_rgb = 0

    forehead_valid = 0
    left_valid = 0
    right_valid = 0

    diagnostic_done = False

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp_ms = int(processed * 1000 / fps)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            processed += 1

            if processed % 100 == 0:
                print(
                    f"Processed {processed}/{total_frames} frames...",
                    flush=True
                )

            if not result.face_landmarks:
                continue

            detected += 1
            landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape

            def get_point(index):
                point = landmarks[index]

                return (
                    int(point.x * w),
                    int(point.y * h)
                )

            # ----------------------------------
            # FOREHEAD ROI
            # ----------------------------------

            forehead_points = [
                get_point(10),
                get_point(151),
                get_point(67),
                get_point(297)
            ]

            # ----------------------------------
            # LEFT CHEEK ROI
            # ----------------------------------

            left_cheek_points = [
                get_point(116),
                get_point(117),
                get_point(118),
                get_point(119)
            ]

            # ----------------------------------
            # RIGHT CHEEK ROI
            # ----------------------------------

            right_cheek_points = [
                get_point(345),
                get_point(346),
                get_point(347),
                get_point(348)
            ]

            # ----------------------------------
            # CREATE ROI RECTANGLES
            # ----------------------------------

            def make_roi(points):
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(w, max(xs))
                y2 = min(h, max(ys))

                return (x1, y1, x2, y2)

            forehead_roi = make_roi(forehead_points)
            left_roi = make_roi(left_cheek_points)
            right_roi = make_roi(right_cheek_points)

            # ----------------------------------
            # RGB EXTRACTION
            # ----------------------------------

            forehead_rgb = extract_rgb(frame, forehead_roi)
            left_rgb = extract_rgb(frame, left_roi)
            right_rgb = extract_rgb(frame, right_roi)

            # ----------------------------------
            # DIAGNOSTIC IMAGE AND HSV ANALYSIS
            # ----------------------------------

            if not diagnostic_done:

                print("\n========== ROI DIAGNOSTIC ==========")
                print("Frame number:", processed)
                print("Frame dimensions:", w, "x", h)

                diagnose_roi(
                    frame,
                    "Forehead",
                    forehead_roi
                )

                diagnose_roi(
                    frame,
                    "Left Cheek",
                    left_roi
                )

                diagnose_roi(
                    frame,
                    "Right Cheek",
                    right_roi
                )

                # Draw ROI rectangles on original frame
                debug_frame = frame.copy()

                for name, roi in [
                    ("Forehead", forehead_roi),
                    ("Left Cheek", left_roi),
                    ("Right Cheek", right_roi)
                ]:
                    x1, y1, x2, y2 = roi

                    cv2.rectangle(
                        debug_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        debug_frame,
                        name,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                image_path = PROJECT_DIR / DIAGNOSTIC_IMAGE

                saved = cv2.imwrite(
                    str(image_path),
                    debug_frame
                )

                if saved:
                    print("\nDiagnostic image saved:", image_path)
                else:
                    print("\nERROR: Could not save diagnostic image.")

                print("\n========== END DIAGNOSTIC ==========\n")

                diagnostic_done = True

            # ----------------------------------
            # COUNT VALID RGB VALUES
            # ----------------------------------

            if forehead_rgb is not None:
                forehead_valid += 1

            if left_rgb is not None:
                left_valid += 1

            if right_rgb is not None:
                right_valid += 1

            # Save only when all three ROIs are valid
            if all(v is not None for v in [
                forehead_rgb,
                left_rgb,
                right_rgb
            ]):

                valid_rgb += 1

                rows.append([
                    processed / fps,
                    *forehead_rgb,
                    *left_rgb,
                    *right_rgb
                ])

    cap.release()

    # ==========================================
    # SAVE CSV
    # ==========================================

    output_path = PROJECT_DIR / OUTPUT_CSV

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "forehead_R", "forehead_G", "forehead_B",
            "left_cheek_R", "left_cheek_G", "left_cheek_B",
            "right_cheek_R", "right_cheek_G", "right_cheek_B"
        ])

        writer.writerows(rows)

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    elapsed = time.time() - start_time

    detection_rate = (
        detected / processed * 100 if processed else 0
    )

    valid_rate = (
        valid_rgb / processed * 100 if processed else 0
    )

    print("\n========== EXTRACTION RESULTS ==========")

    print(f"Processed frames: {processed}")
    print(f"Detected face frames: {detected}")
    print(f"Valid RGB frames: {valid_rgb}")

    print(f"Forehead valid RGB: {forehead_valid}")
    print(f"Left cheek valid RGB: {left_valid}")
    print(f"Right cheek valid RGB: {right_valid}")

    print(f"Face detection rate: {detection_rate:.2f}%")
    print(f"Valid RGB rate: {valid_rate:.2f}%")

    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
