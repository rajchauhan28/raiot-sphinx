"""
ISL Classifier Tester (YOLOv8 Pose + PyTorch)
Author: Raj Singh Chauhan
This script loads your trained model and tests it on a single image.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from train_isl_pytorch import ISLClassifier  # import your classifier class
import argparse

# ===========================
# 1. SETUP
# ===========================
print("🔧 Initializing...")
# Argument parser
parser = argparse.ArgumentParser(description="ISL Classifier Tester")
parser.add_argument("--image", type=str, required=True, help="Path to the test image.")
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ===========================
# 2. LOAD CLASSIFIER
# ===========================
print("\n📦 Loading trained classifier model...")
checkpoint_path = "isl_classifier_yolov8.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
except FileNotFoundError:
    print(f"❌ Error: {checkpoint_path} not found!")
    exit()

classes = checkpoint["classes"]
print(f"✅ Loaded classifier with {len(classes)} classes:")
print(f"   {classes}")

model = ISLClassifier(checkpoint["input_size"], checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Classifier model loaded successfully.")

# ===========================
# 3. LOAD YOLOv8 POSE MODEL
# ===========================
print("\n🤖 Loading YOLOv8 Pose model (for landmark extraction)...")
pose_model = YOLO("yolov8n-pose.pt").to(device)
print("✅ YOLOv8 Pose model ready.")

# ===========================
# 4. SELECT TEST IMAGE
# ===========================
# Change this to any ISL image from your dataset
image_path = args.image

print(f"\n🖼️ Loading test image: {image_path}")
image = cv2.imread(image_path)
if image is None:
    print("❌ Could not read image. Please check the path.")
    exit()

print(f"✅ Image loaded: shape = {image.shape}")

# ===========================
# 5. RUN YOLOv8 INFERENCE
# ===========================
print("\n🔍 Extracting landmarks with YOLOv8 Pose...")
results = pose_model(image)

if not results or len(results[0].keypoints.xy) == 0:
    print("⚠️ No human pose detected in image!")
    exit()

keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
print(f"✅ Extracted {len(keypoints)//3} keypoints × (x, y, conf)")

# ===========================
# 6. CLASSIFY SIGN
# ===========================
print("\n🧠 Running classifier inference...")
features = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(features)
    probabilities = torch.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][pred_idx].item()

print("\n===============================")
print(f"🎯 Predicted Sign: {classes[pred_idx]}")
print(f"📊 Confidence: {confidence*100:.2f}%")
print("===============================")

# Show top-3 predictions
top_probs, top_indices = torch.topk(probabilities, 3)
print("\n🔝 Top 3 Predictions:")
for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
    print(f"  {rank}. {classes[idx]} ({prob.item()*100:.2f}%)")

# ===========================
# 7. DISPLAY RESULT
# ===========================
label = f"{classes[pred_idx]} ({confidence:.2f})"
cv2.putText(image, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
cv2.imshow("ISL Prediction", image)

print("\n👀 Press 'Q' in the image window to exit.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
