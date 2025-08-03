import cv2
import os
import uuid  # For generating random names

# Create folder to save faces
save_path = "recorded_faces"
os.makedirs(save_path, exist_ok=True)

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Faces will be detected and saved automatically. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face and save with random name
        face_img = frame[y:y+h, x:x+w]
        random_name = f"face_{uuid.uuid4().hex}.jpg"  # Unique random name
        filename = os.path.join(save_path, random_name)
        cv2.imwrite(filename, face_img)
        print(f"[INFO] Face saved as: {filename}")

    cv2.imshow("Auto Face Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
