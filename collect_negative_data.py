import cv2
import os
import time

# --- Configuration ---
WEBCAM_INDEX = 0
SAVE_PATH = "Indian/0_background"
IMAGE_COUNT_TARGET = 200

# --- Setup ---
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"✅ Created directory: {SAVE_PATH}")

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"❌ Error: Could not open webcam (index: {WEBCAM_INDEX}).")
    exit()

img_counter = 0
print("\n🚀 Starting data collection...")
print(f"   - Press [SPACE] to capture an image.")
print(f"   - Press [Q] to quit.")

# --- Collection Loop ---
while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Dropped frame.")
        time.sleep(0.1)
        continue

    # Display instructions and counter on the frame
    display_frame = frame.copy()
    text = f"Saved: {img_counter}/{IMAGE_COUNT_TARGET} | Press [SPACE] to capture, [Q] to quit"
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Data Collection - (Press SPACE to Capture)", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '): # Spacebar
        img_name = f"{SAVE_PATH}/background_{time.time()}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"📸 Saved {img_name}")
        img_counter += 1

# --- Cleanup ---
print(f"\n🛑 Stopping... You have collected {img_counter} images.")
cap.release()
cv2.destroyAllWindows()
print("✅ Resources released.")
