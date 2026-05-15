import cv2
import numpy as np
import config
from distance_estimator import DistanceEstimator


class Calibrator:
    
    def __init__(self):
        self.calibrating = False
        self.calibration_points = []
        self.reference_distance = None
        self.reference_width = None
        self.estimator = DistanceEstimator()
        self.calibrated_focal_length = None
        
    def start_calibration(self, reference_distance, reference_width):
        self.calibrating = True
        self.reference_distance = reference_distance
        self.reference_width = reference_width
        self.calibration_points = []
        print(f"Place reference object at {reference_distance}m")
        print("Press 'C' when object is in position")
        
    def process_frame(self, frame, detections):
        if not self.calibrating:
            return frame
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        instructions = [
            "CALIBRATION MODE",
            f"Reference: {self.reference_width}m wide object at {self.reference_distance}m",
            "",
            "1. Place reference object in frame",
            "2. Ensure it's clearly visible",
            "3. Press SPACE to capture",
            "4. Or select from auto-detected objects",
            "",
            "Press ESC to cancel calibration"
        ]
        
        y_pos = 50
        for instruction in instructions:
            cv2.putText(display, instruction, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255), 2)
            y_pos += 35
        
        for i, det in enumerate(detections):
            x, y, bw, bh = det['bbox']
            class_name = det['class']
            
            color = (0, 255, 0)
            cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 3)
            
            label = f"[{i+1}] {class_name}"
            cv2.putText(display, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return display
    
    def capture_calibration(self, detection_index, detections):
        if detection_index < 0 or detection_index >= len(detections):
            return False, None
        
        det = detections[detection_index]
        _, _, pixel_width, _ = det['bbox']
        
        if pixel_width == 0:
            return False, None
        
        focal_length = self.estimator.calibrate_focal_length(
            self.reference_distance, 
            self.reference_width, 
            pixel_width
        )
        
        self.calibrated_focal_length = focal_length
        self.calibration_points.append({
            'pixel_width': pixel_width,
            'focal_length': focal_length
        })
        
        print(f"Focal length calculated: {focal_length:.2f}")
        
        return True, focal_length
    
    def manual_calibration(self, pixel_width):
        if self.reference_distance is None or self.reference_width is None:
            return None
        
        focal_length = (pixel_width * self.reference_distance) / self.reference_width
        self.calibrated_focal_length = focal_length
        
        return focal_length
    
    def finish_calibration(self):
        self.calibrating = False
        
        if self.calibration_points:
            avg_focal = np.mean([p['focal_length'] for p in self.calibration_points])
            print(f"Calibration complete. Average focal length: {avg_focal:.2f}")
            return avg_focal
        
        return self.calibrated_focal_length
    
    def cancel_calibration(self):
        self.calibrating = False
        self.calibration_points = []
        print("Calibration cancelled")
        
    def get_calibration_ui_state(self):
        return {
            'calibrating': self.calibrating,
            'points_captured': len(self.calibration_points),
            'focal_length': self.calibrated_focal_length
        }


class MultiPointCalibrator(Calibrator):
    
    def __init__(self):
        super().__init__()
        self.distance_points = []  # (distance, pixel_width) pairs
        
    def add_measurement(self, distance, pixel_width):
        self.distance_points.append({
            'distance': distance,
            'pixel_width': pixel_width
        })
        
    def compute_calibration_curve(self):
        if len(self.distance_points) < 2:
            return None
        
        distances = [p['distance'] for p in self.distance_points]
        pixel_widths = [p['pixel_width'] for p in self.distance_points]
        
        inv_distances = [1/d for d in distances]
        
        n = len(distances)
        sum_x = sum(inv_distances)
        sum_y = sum(pixel_widths)
        sum_xy = sum(x*y for x, y in zip(inv_distances, pixel_widths))
        sum_x2 = sum(x*x for x in inv_distances)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if self.reference_width:
            focal_length = slope / self.reference_width
            return focal_length
        
        return slope