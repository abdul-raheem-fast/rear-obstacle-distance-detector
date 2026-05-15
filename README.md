# Rear Obstacle Distance Detector 🚗🚦

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Course_Project-orange.svg)

A real-time Computer Vision system designed to assist drivers during reverse parking. The system detects rear obstacles, estimates their distance, and provides visual and auditory alerts to prevent rear-end collisions.

## 👥 Group Members
- **M. Abdul Raheem**
- **Nouman Hassan**
- **M. Sohaib**

## ✨ Features
- **Real-Time Object Detection**: Utilizes deep learning models (MobileNet SSD / YOLO) to detect vehicles, pedestrians, and obstacles.
- **Accurate Distance Estimation**: Calculates the approximate distance from the camera to the detected obstacles.
- **Visual GUI & HUD**: Displays bounding boxes, distance metrics, and safety zones.
- **Auditory Warnings**: Beeps based on proximity (the closer the obstacle, the more urgent the sound).
- **Night Mode Enhancement**: Automatically adjusts brightness and contrast for low-light conditions.
- **Camera Calibration**: Ensures distance accuracy by correcting camera distortion.

## 🛠️ Technologies & Libraries
- **Python**
- **OpenCV**: Core computer vision and image processing.
- **NumPy & SciPy**: Matrix operations and distance calculations.
- **Pygame & Playsound**: Audio feedback management.
- **Imutils**: Basic image processing functions.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdul-raheem-fast/rear-obstacle-distance-detector.git
   cd rear-obstacle-distance-detector
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models:**
   Run the model downloader script to fetch required Caffe/YOLO weights.
   ```bash
   python download_models.py
   ```

## 🎯 Usage

Run the main application:
```bash
python main.py
```

*Press `q` to quit the application window.*

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.