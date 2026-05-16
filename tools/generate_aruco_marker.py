import argparse
import os
import cv2
import numpy as np


def get_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not available. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_marker(dictionary, marker_id, side_px, output_path):
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    else:
        marker = np.zeros((side_px, side_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, side_px, marker, 1)
    cv2.imwrite(output_path, marker)


def main():
    parser = argparse.ArgumentParser(description="Generate an ArUco marker image.")
    parser.add_argument("--dict", default="DICT_5X5_100", help="ArUco dictionary name")
    parser.add_argument("--id", type=int, default=0, help="Marker ID")
    parser.add_argument("--side-px", type=int, default=600, help="Marker side length in pixels")
    parser.add_argument("--out", default="aruco_5x5_100_id0.png", help="Output image path")
    args = parser.parse_args()

    dictionary = get_dictionary(args.dict)
    out_path = os.path.abspath(args.out)
    generate_marker(dictionary, args.id, args.side_px, out_path)
    print(f"Saved ArUco marker: {out_path}")
    print("Print this image at the intended real-world size (e.g., 5 cm side).")


if __name__ == "__main__":
    main()
