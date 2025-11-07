import cv2
import torch
import mediapipe as mp
import numpy as np
from train_isl_pytorch import ISLClassifier # Import the classifier class

# --- Configuration ---
WEBCAM_INDEX = 0
MODEL_PATH = "trained_models/isl_classifier_mediapipe.pth"
CONFIDENCE_THRESHOLD = 0.8 # Set a threshold to avoid flickering predictions

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Load Classifier Model ---
print(f"📦 Loading model from {MODEL_PATH}...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]
    input_size = checkpoint["input_size"]
    
    model = ISLClassifier(input_size=input_size, num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded successfully. Classes: {len(classes)}")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")
    exit()

# --- MediaPipe Setup ---
print("🔧 Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, # We trained on single hands, so predict on one
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
print("✅ MediaPipe Hands ready.")

# --- Initialize Webcam ---
print(f"\n📹 Opening webcam (index: {WEBCAM_INDEX})...")
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()
print("✅ Webcam opened successfully.")

# --- Real-time Prediction Loop ---
print("\n🚀 Starting real-time prediction. Press 'q' to quit.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Dropped frame.")
        continue

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB for MediaPipe.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # Draw the hand annotations and classify
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            # --- Classification ---
            # 1. Extract landmarks and flatten
            landmarks_flat = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            # 2. Convert to tensor and send to device
            features = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 3. Predict
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, dim=1)
            
            # 4. Display prediction
            if confidence.item() > CONFIDENCE_THRESHOLD:
                predicted_sign = classes[pred_idx.item()]
                label = f"{predicted_sign} ({confidence.item()*100:.1f}%)"
                
                # Get the bounding box of the hand to place the text
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                cv2.putText(frame, label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("ISL Real-time Classification", frame)

    # Exit condition
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("\n🛑 Stopping...")
hands.close()
cap.release()
cv2.destroyAllWindows()
print("✅ Resources released.")
