
import cv2
import os
import json

# Create required folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# Load Haar Cascade
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade XML file.")

# Get person details
person_id = int(input("Enter Person ID: "))
person_name = input("Enter Person Name: ").strip()

if not person_name:
    raise ValueError("Name cannot be empty.")

# Load existing names
names_file = "trainer/names.json"

if os.path.exists(names_file):
    with open(names_file, "r") as file:
        names = json.load(file)
else:
    names = {}

# Store name against ID
names[str(person_id)] = person_name

with open(names_file, "w") as file:
    json.dump(names, file, indent=4)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

count = 0
max_images = 100

print("Look at the camera. Capturing face images...")

while count < max_images:

    ret, frame = camera.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Capture only when exactly one face is detected
    if len(faces) == 1:

        x, y, w, h = faces[0]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        count += 1

        image_name = f"User.{person_id}.{count}.jpg"

        cv2.imwrite(
            os.path.join("dataset", image_name),
            face
        )

        cv2.putText(
            frame,
            f"Images: {count}/{max_images}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    elif len(faces) > 1:
        cv2.putText(
            frame,
            "Only one person should be visible",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Face registration completed.")
print("Total images captured:", count)
