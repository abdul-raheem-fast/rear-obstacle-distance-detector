import cv2

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

SAFE_DISTANCE = 2.0
CAUTION_DISTANCE = 1.0
DANGER_DISTANCE = 0.5

KNOWN_OBJECTS = {
    "person": 0.45,
    "car": 1.8,
    "bicycle": 0.6,
    "motorcycle": 0.8,
    "bus": 2.5,
    "truck": 2.4,
    "dog": 0.5,
    "cat": 0.3,
    "bottle": 0.08,
    "default": 0.5
}

# Camera-specific focal length (in pixels). Update with calibration per device.
FOCAL_LENGTH_LAPTOP = 800
FOCAL_LENGTH_MOBILE = 800
FOCAL_LENGTH = FOCAL_LENGTH_LAPTOP

# ArUco auto-calibration (optional)
ARUCO_AUTO_CALIBRATION = True
ARUCO_MARKER_SIZE_M = 0.05
ARUCO_DICTIONARY = "DICT_5X5_100"
ARUCO_CALIBRATION_COOLDOWN_S = 3.0

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
OVERLAY_OPACITY = 0.7

ENABLE_AUDIO_WARNINGS = True
WARNING_COOLDOWN = 2.0

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# Limit detections to realistic classes for this project.
# Set to None to allow all model classes (not recommended).
RELEVANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "bus",
    "motorbike",
    "chair",
    "sofa",
    "diningtable",
    "tvmonitor",
    "bottle",
    "pottedplant"
}

COLORS = {
    "safe": (0, 255, 0),
    "caution": (0, 255, 255),
    "danger": (0, 0, 255),
    "text": (255, 255, 255),
    "overlay": (0, 0, 0),
    "grid": (50, 50, 50)
}

NIGHT_MODE_BRIGHTNESS = 1.5
NIGHT_MODE_CONTRAST = 1.3
NIGHT_MODE_GAMMA = 0.8

GRID_LINES = 5