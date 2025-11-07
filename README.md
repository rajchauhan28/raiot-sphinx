# Indian Sign Language (ISL) Gesture Recognition

This project provides a deep learning-based solution to classify static Indian Sign Language gestures from images and live webcam feeds. It uses YOLOv8-Pose for landmark extraction and a custom PyTorch MLP for classification.

## Features

-   **Static Image Prediction**: Classify an ISL gesture from a single image file.
-   **Real-time Webcam Prediction**: Classify ISL gestures in real-time using a webcam.
-   **Simple & Powerful Architecture**: Combines the strength of YOLOv8 for pose estimation with a lightweight PyTorch classifier.

## Model Architecture

The system employs a two-stage pipeline for gesture recognition:

1.  **Pose Landmark Extraction**: The pre-trained `yolov8n-pose.pt` model detects a person in an image and extracts 17 key body-pose landmarks.
2.  **Gesture Classification**: A custom Multi-Layer Perceptron (MLP), defined in `train_isl_pytorch.py`, takes the flattened landmark data (a 51-point vector) and classifies it into one of the trained ISL gestures.

For a more detailed explanation of the model, see [model.md](./model.md).

## Setup and Installation

### 1. Clone the Repository (with Git LFS)

This repository uses Git LFS (Large File Storage) to handle large model files (`.pt`, `.pth`).

First, install Git LFS on your system. You can find instructions at [git-lfs.github.com](https://git-lfs.github.com/).

After installing, enable it and clone the repository:
```bash
# Install Git LFS (only needs to be done once per user account)
git lfs install

# Clone the repository
git clone https://github.com/your-username/RAIoT_ai.git
cd RAIoT_ai
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the classifier on your own dataset:

1.  **Prepare the dataset**: Organize your images in the `Indian/` directory, with one subdirectory for each gesture class (e.g., `Indian/A/`, `Indian/B/`, etc.).
2.  **Run the training script**:
    ```bash
    python train_isl_pytorch.py
    ```
3.  The script will process the dataset, train the MLP classifier, and save the resulting model to `trained_models/isl_classifier_yolov8.pth`.

### Predicting from an Image

To classify a gesture from a single image file:

```bash
python yolov8n-pose-predict.py --image /path/to/your/image.jpg
```
The script will display the image with the predicted gesture and confidence score.

### Live Prediction from Webcam

To start real-time gesture classification using your webcam:

```bash
python live_predict.py
```
Press the 'q' key to exit the live feed.

## Key Files in this Repository

-   `train_isl_pytorch.py`: Script to train the gesture classifier.
-   `yolov8n-pose-predict.py`: Script to classify a gesture from a single image.
-   `live_predict.py`: Script for real-time gesture classification from a webcam.
-   `model.md`: Detailed documentation of the model architecture.
-   `yolov8n-pose.pt`: The pre-trained YOLOv8 pose estimation model weights (tracked with Git LFS).
-   `isl_classifier_yolov8.pth`: The pre-trained ISL classifier model weights (tracked with Git LFS).
-   `Indian/`: The directory containing the image dataset for training.
