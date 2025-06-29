# Dinesh-Pandian.K-
Blood group detection using fingerprint 
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Example labels: 'A', 'B', 'AB', 'O'
LABELS = ['A', 'B', 'AB', 'O']

def load_dataset(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            features = extract_features(img)
            images.append(features)
            labels.append(label)
    return np.array(images), np.array(labels)

def extract_features(img):
    # Basic features: flatten pixel values (for now)
    # You can enhance this with ridge frequency, minutiae, etc.
    return img.flatten()

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# Load your dataset (organize folders like: dataset/A, dataset/B, etc.)
data_dir = 'fingerprint_dataset'  # <-- update this with your dataset path
X, y = load_dataset(data_dir)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict on new fingerprint
def predict_new_fingerprint(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    features = extract_features(img).reshape(1, -1)
    prediction = model.predict(features)[0]
    print(f"Predicted Blood Group: {prediction}")

# Example usage
# predict_new_fingerprint("test_fingerprint.png")
