
import cv2
import os
import json
import csv
from datetime import datetime

# Check model and names files
model_path = "trainer/face_model.yml"
names_path = "trainer/names.json"

if not os.path.exists(model_path):
    raise FileNotFoundError("Train the model first.")

if not os.path.exists(names_path):
    raise FileNotFoundError("Names file not found.")

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load registered names
with open(names_path, "r") as file:
    names = json.load(file)

# Load Haar Cascade
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade XML file.")

# Create attendance folder
os.makedirs("attendance", exist_ok=True)

# Create today's CSV file
today = datetime.now().strftime("%Y-%m-%d")
csv_file = f"attendance/attendance_{today}.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Person ID",
            "Name",
            "Date",
            "Time",
            "Status"
        ])

# Read already marked attendance
marked_ids = set()

with open(csv_file, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        marked_ids.add(row["Person ID"])

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

# Starting threshold; needs testing and tuning
threshold = 70

print("Attendance system started.")
print("Press Q to exit.")

while True:

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

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        person_id, distance = recognizer.predict(face)

        if distance < threshold and str(person_id) in names:

            name = names[str(person_id)]
            label = name
            color = (0, 255, 0)

            if str(person_id) not in marked_ids:

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(csv_file, "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        person_id,
                        name,
                        date,
                        time,
                        "Present"
                    ])

                marked_ids.add(str(person_id))

                print("Attendance marked for", name)

        else:
            label = "Unknown"
            color = (0, 0, 255)

        # Draw rectangle and name
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")
