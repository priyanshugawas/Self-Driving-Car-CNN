# Self-Driving Car CNN Project

This repository contains an end-to-end Convolutional Neural Network
(CNN) project for autonomous driving based on behavioral cloning. The
model is trained on image--steering angle pairs to predict steering
commands directly from camera input.

------------------------------------------------------------------------

## 1. Project Overview

The goal of this project is to build, train, evaluate, and deploy a CNN
capable of predicting steering angles from road images captured by a
front-facing camera mounted on a vehicle. This project is inspired by
early NVIDIA behavioral cloning approaches.

Key components include:

-   Dataset ingestion and preprocessing
-   Image augmentation pipeline
-   CNN architecture for regression
-   Training loop with checkpoints and logging
-   Inference pipeline for real-time steering predictions
-   Model export for deployment

------------------------------------------------------------------------

## 2. Dataset

The dataset used in this project consists of center-view driving images
paired with steering angles.

**Dataset Size:** 2.2 GB (Too large to upload to GitHub)

**Dataset Link:**\
https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view

------------------------------------------------------------------------

## 3. Project Structure

    self-driving-car-cnn/
    │
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── data/
    │   ├── driving_dataset/
    │   │   ├── data.txt
    │   │   ├── 0.jpg
    │   │   ├── ...
    │   └── steering_wheel_image.jpg
    │
    ├── src/
    │   ├── data/
    │   │   └── driving_data.py
    │   │
    │   ├── model/
    │   │   └── nvidia_model.py
    │   │
    │   ├── training/
    │   │   └── train.py
    │   │
    │   ├── inference/
    │   │   ├── run_dataset.py
    │   │   └── run_webcam.py
    │
    ├── save/
    │   └── (Will be auto-created during training)
    │
    └── logs/
        └── (Created automatically by training)


------------------------------------------------------------------------

## 4. Model Architecture

The CNN architecture includes:

-   Input normalization
-   Convolutional layers with ReLU activation
-   Strided convolutions instead of pooling
-   Fully connected layers for regression
-   Linear output neuron predicting steering angle

A dropout layer is optionally added to reduce overfitting.

------------------------------------------------------------------------

## 5. Training

To train the model:

    python src/train.py

Training features include:

-   Data augmentation
-   Learning rate scheduling
-   Model checkpointing
-   TensorBoard logging

------------------------------------------------------------------------

## 6. Inference

To run predictions on a test image:

    python src/inference.py --image path/to/image.jpg

This returns a floating-point steering value.

------------------------------------------------------------------------

## 7. Requirements

Install dependencies:

    pip install -r requirements.txt

The project requires TensorFlow/Keras, NumPy, OpenCV, and Matplotlib,
among others.

------------------------------------------------------------------------

## 8. Notes

-   The dataset is **not included** in this repository due to its size
    (2.2 GB).
-   Isert your dataset download link in the placeholder above.
-   To train the model, ensure the dataset is placed inside `data/driving_dataset/`.

------------------------------------------------------------------------

## 9. License

This project is open-source and free to use for educational or research purposes.

------------------------------------------------------------------------

## 10. Author

Priyanshu S. Gawas
