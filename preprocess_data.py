import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# --- MediaPipe Hand Extractor (self-contained in this script) ---
class MediaPipeHandExtractor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5)
        self.num_landmarks = 21

    def extract_landmarks(self, image: np.ndarray):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return np.zeros(self.num_landmarks * 3, dtype=np.float32)
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_flat = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return landmarks_flat

    def close(self):
        self.hands.close()

# --- Main Pre-processing Logic ---
def create_preprocessed_dataset(data_path, output_path):
    print("🚀 Starting dataset pre-processing...")
    extractor = MediaPipeHandExtractor()
    
    all_features = []
    all_labels = []
    
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"🔍 Found {len(classes)} classes: {classes}")

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        image_files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        print(f"\nProcessing class: {cls} ({len(image_files)} images)")
        for img_name in tqdm(image_files, desc=f"  -> {cls}"):
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"⚠️ Warning: Could not read image {img_path}. Skipping.")
                continue
                
            landmarks = extractor.extract_landmarks(image)
            all_features.append(landmarks)
            all_labels.append(class_to_idx[cls])

    extractor.close()
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    # Save to file
    torch.save({
        'features': features_tensor,
        'labels': labels_tensor,
        'classes': classes
    }, output_path)
    
    print(f"\n✅ Pre-processing complete!")
    print(f"   - Total samples: {len(all_features)}")
    print(f"   - Features tensor shape: {features_tensor.shape}")
    print(f"   - Labels tensor shape: {labels_tensor.shape}")
    print(f"   - Data saved to: {output_path}")

if __name__ == "__main__":
    DATASET_DIR = "/home/reign/ddrive/RAIoT_ai/Indian"
    OUTPUT_FILE = "preprocessed_data.pt"
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory not found at {DATASET_DIR}")
    else:
        create_preprocessed_dataset(DATASET_DIR, OUTPUT_FILE)
