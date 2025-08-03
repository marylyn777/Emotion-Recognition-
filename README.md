# Emotion-Recognition-CNN
A lightweight CNN-based system for real-time facial emotion recognition using the FER2013 dataset. The model classifies faces into seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
Features:
1) Trained on FER2013 (35,887 grayscale 48×48 images)
2) Built with PyTorch, Gradio, and OpenCV
3) 58.86% accuracy on the test set

Supports:
1) Image and video uploads
2) Real-time webcam inference
3) Simple GUI for demonstration and educational use

Notes:
1) Webcam and upload functionalities are separated due to environment constraints
2) Confusion observed between similar emotions (e.g., sad vs. neutral)


