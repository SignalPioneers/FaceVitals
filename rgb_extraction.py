
import cv2
import mediapipe as mp
import time
import os
import csv
import numpy as np

# =====================================================
# FACEVITALS - RGB EXTRACTION WITH SKIN MASKING
# =====================================================

MODEL_PATH = "face_landmarker.task"
OUTPUT_FILE = "rgb_signals.csv"

# HSV skin detection thresholds.
# These are initial experimental thresholds and may need tuning.
LOWER_SKIN_1 = np.array([0, 20, 40], dtype=np.uint8)
UPPER_SKIN_1 = np.array([25, 200, 255], dtype=np.uint8)

LOWER_SKIN_2 = np.array([160, 20, 40], dtype=np.uint8)
UPPER_SKIN_2 = np.array([180, 200, 255], dtype=np.uint8)

MIN_SKIN_RATIO = 0.10


# =====================================================
# 1. LOAD MEDIAPIPE MODEL
# =====================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Face Landmarker model not found!")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


# =====================================================
# 2. OPEN WEBCAM
# =====================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

start_time = time.monotonic()


# =====================================================
# 3. PREPARE CSV FILE
# KEEPING THE ORIGINAL COLUMN NAMES
# =====================================================

csv_file = open(OUTPUT_FILE, "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "Time",
    "Forehead_R", "Forehead_G", "Forehead_B",
    "LeftCheek_R", "LeftCheek_G", "LeftCheek_B",
    "RightCheek_R", "RightCheek_G", "RightCheek_B"
])


# =====================================================
# 4. SKIN MASKING AND RGB EXTRACTION FUNCTION
# =====================================================

def extract_rgb(frame, roi):

    x1, y1, x2, y2 = roi

    # Ensure ROI coordinates are inside the image.
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None, 0.0

    # Convert the cropped image from BGR to HSV.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Detect possible skin pixels in two hue ranges.
    mask1 = cv2.inRange(
        hsv, LOWER_SKIN_1, UPPER_SKIN_1
    )

    mask2 = cv2.inRange(
        hsv, LOWER_SKIN_2, UPPER_SKIN_2
    )

    # Combine both masks.
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # Calculate the percentage of selected pixels.
    total_pixels = crop.shape[0] * crop.shape[1]
    skin_pixels = cv2.countNonZero(skin_mask)

    skin_ratio = skin_pixels / total_pixels

    # Reject an ROI if too few possible skin pixels are found.
    minimum_pixels = max(20, int(total_pixels * MIN_SKIN_RATIO))

    if skin_pixels < minimum_pixels:
        return None, skin_mask, skin_ratio

    # Calculate average BGR using selected pixels only.
    mean_bgr = cv2.mean(
        crop,
        mask=skin_mask
    )[:3]

    # Convert BGR values to RGB.
    mean_rgb = mean_bgr[::-1]

    return mean_rgb, skin_mask, skin_ratio


# =====================================================
# 5. START FACE DETECTION AND RGB COLLECTION
# =====================================================

valid_frames = 0
rejected_frames = 0

try:

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Could not read camera frame.")
                break

            height, width, _ = frame.shape

            # Convert BGR to RGB for MediaPipe.
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            elapsed_time = time.monotonic() - start_time
            timestamp_ms = int(elapsed_time * 1000)

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            # Create an empty mask for visualization.
            full_skin_mask = np.zeros(
                (height, width), dtype=np.uint8
            )

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # Find face bounding box.
                x_points = [
                    int(p.x * width) for p in landmarks
                ]

                y_points = [
                    int(p.y * height) for p in landmarks
                ]

                x_min = max(0, min(x_points))
                x_max = min(width, max(x_points))

                y_min = max(0, min(y_points))
                y_max = min(height, max(y_points))

                face_width = x_max - x_min
                face_height = y_max - y_min

                if face_width > 0 and face_height > 0:

                    # -------------------------------------
                    # DEFINE THREE FACIAL REGIONS
                    # -------------------------------------

                    forehead = (
                        int(x_min + 0.30 * face_width),
                        int(y_min + 0.08 * face_height),
                        int(x_min + 0.70 * face_width),
                        int(y_min + 0.27 * face_height)
                    )

                    left_cheek = (
                        int(x_min + 0.12 * face_width),
                        int(y_min + 0.40 * face_height),
                        int(x_min + 0.42 * face_width),
                        int(y_min + 0.62 * face_height)
                    )

                    right_cheek = (
                        int(x_min + 0.58 * face_width),
                        int(y_min + 0.40 * face_height),
                        int(x_min + 0.88 * face_width),
                        int(y_min + 0.62 * face_height)
                    )

                    # -------------------------------------
                    # EXTRACT SKIN RGB FROM EACH REGION
                    # -------------------------------------

                    forehead_rgb, forehead_mask, forehead_ratio = (
                        extract_rgb(frame, forehead)
                    )

                    left_rgb, left_mask, left_ratio = (
                        extract_rgb(frame, left_cheek)
                    )

                    right_rgb, right_mask, right_ratio = (
                        extract_rgb(frame, right_cheek)
                    )

                    # -------------------------------------
                    # DISPLAY SELECTED SKIN PIXELS
                    # -------------------------------------

                    for roi, mask in [
                        (forehead, forehead_mask),
                        (left_cheek, left_mask),
                        (right_cheek, right_mask)
                    ]:

                        if mask is not None:

                            x1, y1, x2, y2 = roi

                            full_skin_mask[y1:y2, x1:x2] = mask

                    # -------------------------------------
                    # SAVE RGB DATA
                    # -------------------------------------

                    if (
                        forehead_rgb is not None
                        and left_rgb is not None
                        and right_rgb is not None
                    ):

                        writer.writerow([
                            elapsed_time,
                            *forehead_rgb,
                            *left_rgb,
                            *right_rgb
                        ])

                        valid_frames += 1

                    else:
                        rejected_frames += 1

                    # -------------------------------------
                    # DRAW ROI RECTANGLES
                    # -------------------------------------

                    cv2.rectangle(
                        frame,
                        forehead[:2],
                        forehead[2:],
                        (0, 255, 0),
                        2
                    )

                    cv2.rectangle(
                        frame,
                        left_cheek[:2],
                        left_cheek[2:],
                        (255, 0, 0),
                        2
                    )

                    cv2.rectangle(
                        frame,
                        right_cheek[:2],
                        right_cheek[2:],
                        (0, 0, 255),
                        2
                    )

                    # Display skin selection percentages.
                    cv2.putText(
                        frame,
                        f"Forehead: {forehead_ratio * 100:.1f}%",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Left cheek: {left_ratio * 100:.1f}%",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Right cheek: {right_ratio * 100:.1f}%",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Valid frames: {valid_frames}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

            else:

                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Show camera with ROI rectangles.
            cv2.imshow(
                "FaceVitals - RGB Extraction",
                frame
            )

            # Show pixels selected by the skin mask.
            cv2.imshow(
                "FaceVitals - Skin Mask",
                full_skin_mask
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    camera.release()
    csv_file.close()
    cv2.destroyAllWindows()


# =====================================================
# 6. DISPLAY RECORDING SUMMARY
# =====================================================

print("\n==========================================")
print("       RGB EXTRACTION SUMMARY")
print("==========================================")

print(f"Valid frames: {valid_frames}")
print(f"Rejected frames: {rejected_frames}")
print(f"Total processed frames: {valid_frames + rejected_frames}")

if valid_frames > 0:
    print(f"Data saved in {OUTPUT_FILE}")
else:
    print("WARNING: No valid RGB measurements were collected.")

print("\nRGB extraction completed.")
