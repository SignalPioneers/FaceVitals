import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. Start the webcam (0 is usually the built-in webcam)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam connected successfully! Press 'q' to quit.")

while camera.isOpened():
    success, frame = camera.read()
    if not success:
        print("Could not read camera frame.")
        break

    # Flip the frame horizontally for a mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Convert the BGR image from OpenCV to RGB required by MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to a MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform face landmark detection
    detection_result = detector.detect(mp_image)

    # 3. Draw face landmarks on the frame if any are detected
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Loop through all 468/478 facial landmarks and draw dots on the face
            for landmark in face_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) # Green dots

    # 4. Display the resulting frame in a window
    cv2.imshow('FaceVitals - Face Landmarker', frame)

    # Press 'q' to stop the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Release resources
camera.release()
cv2.destroyAllWindows()
print("Face detection stopped.")