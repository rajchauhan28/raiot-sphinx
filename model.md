# ISL Static Gesture Classification Model

This document outlines the architecture and workflow of the Indian Sign Language (ISL) static gesture classification model.

## Pipeline Architecture

The system uses a two-stage pipeline to classify ISL gestures from a single image:

1.  **Stage 1: Pose Estimation (YOLOv8-Pose)**: A pre-trained YOLOv8 model extracts human pose landmarks from the input image.
2.  **Stage 2: Sign Classification (MLP)**: A Multi-Layer Perceptron (MLP) classifies the gesture based on the extracted landmarks.

---

## Stage 1: Pose Estimation

-   **Model**: `yolov8n-pose.pt`
-   **Function**: Detects a human figure in an image and extracts keypoints for body joints.
-   **Output**: A set of 17 keypoints, where each keypoint consists of `(x, y, confidence)`. This results in a flat feature vector of 51 values (17 keypoints * 3 values).

---

## Stage 2: Sign Classifier

-   **Model Type**: Multi-Layer Perceptron (MLP) built with PyTorch.
-   **Input**: A 51-dimensional feature vector from the YOLOv8-Pose model.
-   **Output**: A probability distribution over the different ISL sign classes.
-   **Architecture**:
    1.  Input Layer: `(Linear -> ReLU -> Dropout)` with `51` input features and `512` output features. Dropout rate is `0.3`.
    2.  Hidden Layer 1: `(Linear -> ReLU -> Dropout)` with `512` input features and `256` output features. Dropout rate is `0.3`.
    3.  Hidden Layer 2: `(Linear -> ReLU -> Dropout)` with `256` input features and `128` output features. Dropout rate is `0.2`.
    4.  Output Layer: `(Linear)` with `128` input features and `num_classes` output features.

---

## Training (`train_isl_pytorch.py`)

The training script orchestrates the process of learning the sign classifier.

-   **Dataset**: The script expects a dataset of images organized into subdirectories, where each subdirectory name corresponds to a class label.
-   **Process**:
    1.  The `YOLOv8PoseExtractor` processes each image in the dataset to extract the 51-feature landmark vector.
    2.  The dataset of landmark vectors is split into an 80% training set and a 20% validation set.
    3.  The MLP classifier is trained on this data using the Adam optimizer and Cross-Entropy loss.
-   **Output**: The script saves the trained model weights and metadata (class labels, input size) to `trained_models/isl_classifier_yolov8.pth`.

---

## Inference (`yolov8n-pose-predict.py`)

This script uses the trained model to perform inference on a new image.

-   **Purpose**: To classify a sign gesture in a single, static image.
-   **Process**:
    1.  Loads the trained `ISLClassifier` from `isl_classifier_yolov8.pth`.
    2.  Loads the `yolov8n-pose.pt` model for landmark extraction.
    3.  Accepts a path to an image via a command-line argument (`--image`).
    4.  Extracts the 51-feature landmark vector from the image.
    5.  Feeds the vector into the classifier to get a prediction.
    6.  Prints the top predicted sign and confidence score to the console.
    7.  Displays the original image with the predicted label overlaid.
-   **Usage**:
    ```bash
    python yolov8n-pose-predict.py --image <path_to_your_image.jpg>
    ```
