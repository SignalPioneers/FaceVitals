
import cv2

# STEP 1: Access the laptop webcam
camera = cv2.VideoCapture(0)

# STEP 2: Check whether the camera opened
if not camera.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam connected successfully!")

# STEP 3: Capture video continuously
while True:

    # Read one frame from the camera
    ret, frame = camera.read()

    # Check if frame was captured successfully
    if not ret:
        print("Error: Could not read video frame.")
        break

    # STEP 4: Display the live video
    cv2.imshow("FaceVitals - Webcam", frame)

    # STEP 5: Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 6: Release the webcam
camera.release()

# STEP 7: Close the video window
cv2.destroyAllWindows()

print("Webcam closed.")