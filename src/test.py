# test.py
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# Paths
MODEL_PATH = "../models/best_model.h5"
CLASS_MAP_PATH = "../models/class_indices.json"
TEST_DIR = "../dataset/test_images/"

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Load class names
with open(CLASS_MAP_PATH, "r") as f:
    class_names = json.load(f)

# Reverse map for easy lookup
idx_to_class = {int(k): v for k, v in class_names.items()}

# Gather test images and labels
test_images = []
true_labels = []

for filename in os.listdir(TEST_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        test_images.append(os.path.join(TEST_DIR, filename))
        # Assuming label is in CSV filename without extension (adjust if needed)
        # Example: 1ae8c165fd53.png → lookup in CSV for diagnosis
        label = int(filename.split("_")[-1].replace(".png",""))  # adjust if needed
        true_labels.append(label)

# Predictions
pred_labels = []
for img_path in test_images:
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    pred_labels.append(pred_class)
    print(f"{os.path.basename(img_path)} → Predicted: {idx_to_class[pred_class]}")

# Evaluation
print("\n=== Classification Report ===")
print(classification_report(true_labels, pred_labels, target_names=list(idx_to_class.values())))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(true_labels, pred_labels))
