
import cv2
import numpy as np
import os

dataset_path = "dataset"

faces = []
ids = []

# Read all images from dataset
for filename in os.listdir(dataset_path):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(dataset_path, filename)

    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    # Extract person ID from filename
    person_id = int(filename.split(".")[1])

    faces.append(image)
    ids.append(person_id)

# Check dataset
if len(faces) == 0:
    raise ValueError("No face images found!")

# Convert IDs to NumPy array
ids = np.array(ids, dtype=np.int32)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the model
recognizer.train(faces, ids)

# Create trainer folder
os.makedirs("trainer", exist_ok=True)

# Save trained model
recognizer.write("trainer/face_model.yml")

print("Model trained successfully!")
print("Total images:", len(faces))
print("Total persons:", len(set(ids)))
