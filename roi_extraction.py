
import cv2
import mediapipe as mp
import time
import os

# Load MediaPipe model
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    print("Face Landmarker model not found!")
    exit()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access webcam.")
    exit()

start_time = time.monotonic()

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Find face bounding box
            x_points = [int(p.x * width) for p in landmarks]
            y_points = [int(p.y * height) for p in landmarks]

            x_min = max(0, min(x_points))
            x_max = min(width, max(x_points))
            y_min = max(0, min(y_points))
            y_max = min(height, max(y_points))

            face_width = x_max - x_min
            face_height = y_max - y_min

            # Forehead ROI
            forehead = (
                int(x_min + 0.30 * face_width),
                int(y_min + 0.08 * face_height),
                int(x_min + 0.70 * face_width),
                int(y_min + 0.27 * face_height)
            )

            # Left cheek ROI (image-left)
            left_cheek = (
                int(x_min + 0.12 * face_width),
                int(y_min + 0.40 * face_height),
                int(x_min + 0.42 * face_width),
                int(y_min + 0.62 * face_height)
            )

            # Right cheek ROI (image-right)
            right_cheek = (
                int(x_min + 0.58 * face_width),
                int(y_min + 0.40 * face_height),
                int(x_min + 0.88 * face_width),
                int(y_min + 0.62 * face_height)
            )

            # Draw face bounding box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (255, 255, 0), 2
            )

            # Draw ROIs
            cv2.rectangle(
                frame, forehead[:2], forehead[2:],
                (0, 255, 0), 2
            )

            cv2.rectangle(
                frame, left_cheek[:2], left_cheek[2:],
                (255, 0, 0), 2
            )

            cv2.rectangle(
                frame, right_cheek[:2], right_cheek[2:],
                (0, 0, 255), 2
            )

            cv2.putText(
                frame, "Forehead",
                (forehead[0], forehead[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Left cheek",
                (left_cheek[0], left_cheek[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2
            )

            cv2.putText(
                frame, "Right cheek",
                (right_cheek[0], right_cheek[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2
            )

        cv2.imshow("FaceVitals - ROI Extraction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
