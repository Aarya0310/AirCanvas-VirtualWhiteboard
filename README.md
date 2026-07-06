# AirCanvas: Virtual Whiteboard

A real-time computer vision application that enables touchless, air-based drawing and interaction using hand gesture recognition.

## 🚀 Overview
**AirCanvas** is a gesture-controlled virtual whiteboard built with Python. By leveraging MediaPipe's hand-tracking capabilities and OpenCV, the system translates hand gestures into digital strokes, allowing users to draw, erase, and select colors in the air.

## 🛠 Tech Stack
* **Language:** Python 3.13
* **Computer Vision:** OpenCV, MediaPipe (Tasks API)
* **Data Processing:** NumPy
* **UI/Web:** Streamlit (for cloud deployment)

## ✨ Features
* **Real-time Tracking:** Smooth finger-tip tracking using MediaPipe's Hand Landmarker model.
* **Dynamic Modes:**
    * **Drawing Mode:** 1 finger up.
    * **Erasing Mode:** 2 fingers up.
* **Color Palette:** Touchless UI buttons for selecting Blue, Green, Red, and White.
* **Canvas Controls:** Built-in "Clear" button for resetting the workspace.
* **Anti-Jitter:** Implemented Exponential Moving Average (EMA) to ensure smooth lines and reduce tracking noise.

## 📂 Project Structure
```text
.
├── .gitignore              # Excludes caches and environment files
├── requirements.txt        # Dependencies
├── app.py                  # Streamlit entry point
├── classes.py              # Core logic & HandDetector class
└── hand_landmarker.task    # Pre-trained ML model