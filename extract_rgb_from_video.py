
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from pathlib import Path

# ==========================================
# FACEVITALS - OFFLINE VIDEO RGB EXTRACTION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "face_landmarker.task"

LOWER_SKIN_1 = np.array([0, 20, 40])
UPPER_SKIN_1 = np.array([25, 200, 255])

LOWER_SKIN_2 = np.array([160, 20, 40])
UPPER_SKIN_2 = np.array([180, 200, 255])

MIN_SKIN_RATIO = 0.10

COLUMNS = [
    "Time",
    "Forehead_R", "Forehead_G", "Forehead_B",
    "LeftCheek_R", "LeftCheek_G", "LeftCheek_B",
    "RightCheek_R", "RightCheek_G", "RightCheek_B"
]


def extract_rgb(frame, roi):

    x1, y1, x2, y2 = roi

    height, width = frame.shape[:2]

    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv, LOWER_SKIN_1, UPPER_SKIN_1
    )

    mask2 = cv2.inRange(
        hsv, LOWER_SKIN_2, UPPER_SKIN_2
    )

    skin_mask = cv2.bitwise_or(mask1, mask2)

    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = crop.shape[0] * crop.shape[1]

    if skin_pixels < max(20, MIN_SKIN_RATIO * total_pixels):
        return None

    mean_bgr = cv2.mean(crop, mask=skin_mask)

    # Convert BGR to RGB
    return (
        mean_bgr[2],
        mean_bgr[1],
        mean_bgr[0]
    )


def main():

    print("\n======================================")
    print(" FACEVITALS OFFLINE RGB EXTRACTION")
    print("======================================\n")

    if not MODEL_PATH.exists():
        print("ERROR: face_landmarker.task not found!")
        return

    video_input = input(
        "Enter video filename (default: baseline.mp4): "
    ).strip()

    if not video_input:
        video_input = "baseline.mp4"

    video_path = Path(video_input)

    if not video_path.is_absolute():
        video_path = PROJECT_DIR / video_path

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    output_input = input(
        "Enter output CSV filename (default: rgb_signals.csv): "
    ).strip()

    if not output_input:
        output_input = "rgb_signals.csv"

    output_path = Path(output_input)

    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path

    if output_path.resolve() == video_path.resolve():
        print("ERROR: Output cannot overwrite the input video.")
        return

    camera = cv2.VideoCapture(str(video_path))

    if not camera.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = camera.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        camera.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0 or total_frames <= 0:
        print("ERROR: Invalid video properties.")
        camera.release()
        return

    print(f"\nVideo: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.2f} seconds")
    print(f"Output: {output_path.name}\n")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    processed_frames = 0
    detected_faces = 0
    valid_rgb_frames = 0
    start_time = time.time()

    try:

        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            with open(
                output_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as csvfile:

                writer = csv.writer(csvfile)
                writer.writerow(COLUMNS)

                while True:

                    success, frame = camera.read()

                    if not success:
                        break

                    frame_time = processed_frames / fps

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    timestamp_ms = int(
                        round(frame_time * 1000)
                    )

                    results = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    values = [np.nan] * 9

                    if results.face_landmarks:

                        detected_faces += 1

                        landmarks = results.face_landmarks[0]

                        height, width = frame.shape[:2]

                        x_coords = [
                            landmark.x for landmark in landmarks
                        ]

                        y_coords = [
                            landmark.y for landmark in landmarks
                        ]

                        x_min = max(
                            0, int(min(x_coords) * width)
                        )

                        x_max = min(
                            width, int(max(x_coords) * width)
                        )

                        y_min = max(
                            0, int(min(y_coords) * height)
                        )

                        y_max = min(
                            height, int(max(y_coords) * height)
                        )

                        face_width = x_max - x_min
                        face_height = y_max - y_min

                        if face_width > 0 and face_height > 0:

                            forehead_roi = (
                                x_min + 0.30 * face_width,
                                y_min + 0.08 * face_height,
                                x_min + 0.70 * face_width,
                                y_min + 0.27 * face_height
                            )

                            left_cheek_roi = (
                                x_min + 0.12 * face_width,
                                y_min + 0.40 * face_height,
                                x_min + 0.42 * face_width,
                                y_min + 0.62 * face_height
                            )

                            right_cheek_roi = (
                                x_min + 0.58 * face_width,
                                y_min + 0.40 * face_height,
                                x_min + 0.88 * face_width,
                                y_min + 0.62 * face_height
                            )

                            forehead_rgb = extract_rgb(
                                frame, forehead_roi
                            )

                            left_rgb = extract_rgb(
                                frame, left_cheek_roi
                            )

                            right_rgb = extract_rgb(
                                frame, right_cheek_roi
                            )

                            if (
                                forehead_rgb is not None
                                and left_rgb is not None
                                and right_rgb is not None
                            ):

                                values = (
                                    list(forehead_rgb)
                                    + list(left_rgb)
                                    + list(right_rgb)
                                )

                                valid_rgb_frames += 1

                    writer.writerow(
                        [round(frame_time, 6)] + values
                    )

                    processed_frames += 1

                    if processed_frames % 100 == 0:

                        percentage = (
                            processed_frames / total_frames
                        ) * 100

                        print(
                            f"Progress: {percentage:.1f}% "
                            f"({processed_frames}/{total_frames})"
                        )

    except Exception as error:

        print(f"\nPROCESSING ERROR: {error}")

    finally:

        camera.release()

    elapsed = time.time() - start_time

    print("\n======================================")
    print(" EXTRACTION SUMMARY")
    print("======================================")

    print(f"Processed frames: {processed_frames}")
    print(f"Detected face frames: {detected_faces}")
    print(f"Valid RGB frames: {valid_rgb_frames}")

    if processed_frames > 0:

        print(
            f"Face detection rate: "
            f"{detected_faces / processed_frames * 100:.2f}%"
        )

        print(
            f"Valid RGB rate: "
            f"{valid_rgb_frames / processed_frames * 100:.2f}%"
        )

    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"CSV saved to: {output_path}")

    print("\nOFFLINE RGB EXTRACTION FINISHED.")


if __name__ == "__main__":
    main()
