"""
ISL Image Dataset Trainer (from Pre-processed MediaPipe Data)

This script trains an ISL classifier using a pre-processed dataset of
MediaPipe hand landmarks, making the training process much faster.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn, optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===============================
# DEVICE CONFIGURATION
# ===============================

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("✅ Using CPU.")

# ===============================
# CLASSIFIER MODEL
# ===============================

class ISLClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# TRAIN FUNCTION
# ===============================

def train_from_preprocessed(data_path, epochs=25, batch_size=128, lr=0.001):
    print(f"💿 Loading pre-processed data from {data_path}...")
    try:
        data = torch.load(data_path)
        features = data['features']
        labels = data['labels']
        classes = data['classes']
    except FileNotFoundError:
        sys.exit(f"❌ Error: Pre-processed data file not found at {data_path}. Please run the pre-processing script first.")
    
    print("✅ Data loaded successfully.")
    
    input_size = features.shape[1]
    num_classes = len(classes)
    
    print(f"   - Input features shape: {features.shape}")
    print(f"   - Number of classes: {num_classes}")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ISLClassifier(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"🚀 Training classifier for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for features_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (preds == labels_batch).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                outputs = model(features_batch)
                _, preds = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (preds == labels_batch).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

    # Save model
    save_path = "trained_models/isl_classifier_mediapipe.pth"
    os.makedirs("trained_models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "input_size": input_size,
        "num_classes": num_classes
    }, save_path)

    print(f"✅ Model saved: {save_path}")

# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    PREPROCESSED_FILE = "preprocessed_data.pt"
    train_from_preprocessed(PREPROCESSED_FILE, epochs=30, batch_size=128)
