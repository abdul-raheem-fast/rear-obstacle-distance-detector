import cv2
import numpy as np
import config


class DistanceEstimator:
    
    def __init__(self, focal_length=config.FOCAL_LENGTH):
        # Note: distance is an approximation based on object width and simple perspective.
        self.focal_length = focal_length
        self.known_objects = config.KNOWN_OBJECTS
        
    def calculate_distance(self, object_name, pixel_width):
        if pixel_width == 0:
            return float('inf')
            
        known_width = self.known_objects.get(object_name.lower(), 
                                            self.known_objects["default"])
        distance = (known_width * self.focal_length) / pixel_width
        return round(distance, 2)
    
    def get_distance_category(self, distance):
        if distance >= config.SAFE_DISTANCE:
            return "SAFE", config.COLORS["safe"]
        elif distance >= config.CAUTION_DISTANCE:
            return "CAUTION", config.COLORS["caution"]
        else:
            return "DANGER", config.COLORS["danger"]
    
    def calibrate_focal_length(self, known_distance, known_width, pixel_width):
        self.focal_length = (pixel_width * known_distance) / known_width
        return self.focal_length
    
    def estimate_distance_from_bbox(self, bbox, object_name):
        _, _, w, _ = bbox
        return self.calculate_distance(object_name, w)


class AdvancedDistanceEstimator(DistanceEstimator):
    
    def __init__(self, focal_length=config.FOCAL_LENGTH):
        super().__init__(focal_length)
        self.reference_points = []
        self.ground_plane_estimated = False
        
    def estimate_distance_with_ground_plane(self, bbox, frame_shape, object_name="default"):
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        bottom_y = y + h
        normalized_position = bottom_y / frame_h
        width_distance = self.calculate_distance(object_name, w)
        
        if width_distance < 3.0:
            estimated_distance = width_distance
        else:
            perspective_factor = 1.0 / (normalized_position + 0.1)
            estimated_distance = (width_distance * 0.7) + (perspective_factor * 0.3)
        
        return round(estimated_distance, 2)
    
    def add_reference_point(self, pixel_coords, real_distance):
        self.reference_points.append({
            'pixel': pixel_coords,
            'distance': real_distance
        })
    
    def calculate_average_distance(self, detections):
        if not detections:
            return None
            
        distances = []
        for det in detections:
            dist = self.estimate_distance_from_bbox(det['bbox'], det['class'])
            distances.append(dist)
        
        distances.sort()
        mid = len(distances) // 2
        return distances[mid] if len(distances) % 2 == 1 else (distances[mid-1] + distances[mid]) / 2