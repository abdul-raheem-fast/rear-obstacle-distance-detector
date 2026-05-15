import cv2
import numpy as np
import time
import config


class ParkingGUIDisplay:
    
    def __init__(self, width=config.DISPLAY_WIDTH, height=config.DISPLAY_HEIGHT):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        
        self.last_warning_time = 0
        self.warning_cooldown = config.WARNING_COOLDOWN
        self.pulse_phase = 0
        
    def draw_interface(self, frame, detections, distances, system_status):
        display = frame.copy()
        display = self._draw_parking_grid(display)
        display = self._draw_detections(display, detections, distances)
        display = self._draw_info_panel(display, system_status)
        display = self._draw_distance_bars(display, detections, distances)
        
        min_distance = min(distances.values()) if distances else float('inf')
        if min_distance < config.DANGER_DISTANCE:
            display = self._draw_danger_warning(display, min_distance)
        elif min_distance < config.CAUTION_DISTANCE:
            display = self._draw_caution_warning(display, min_distance)
        
        display = self._draw_header(display)
        return display
    
    def _draw_parking_grid(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        cv2.line(overlay, (w//2, 0), (w//2, h), config.COLORS["grid"], 1)
        
        zone_positions = [0.3, 0.5, 0.7]
        for pos in zone_positions:
            y = int(h * pos)
            cv2.line(overlay, (0, y), (w, y), config.COLORS["grid"], 1)
            
        center_x = w // 2
        bottom_y = h
        
        pts_left = np.array([
            [center_x - 50, bottom_y],
            [center_x - 150, bottom_y - 100],
            [center_x - 200, bottom_y - 200]
        ], np.int32)
        cv2.polylines(overlay, [pts_left], False, config.COLORS["grid"], 2)
        
        pts_right = np.array([
            [center_x + 50, bottom_y],
            [center_x + 150, bottom_y - 100],
            [center_x + 200, bottom_y - 200]
        ], np.int32)
        cv2.polylines(overlay, [pts_right], False, config.COLORS["grid"], 2)
        
        return overlay
    
    def _draw_detections(self, frame, detections, distances):
        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            obj_id = f"{class_name}_{x}_{y}"
            distance = distances.get(obj_id, float('inf'))
            _, color = self._get_distance_info(distance)
            
            thickness = 3 if distance < config.DANGER_DISTANCE else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            label = f"{class_name.upper()}: {distance:.2f}m ({confidence:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            
            cv2.rectangle(frame, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            cv2.rectangle(frame, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, 2)
            
            cv2.putText(frame, label, (x + 5, y - 5), 
                       self.font, 0.6, config.COLORS["text"], 2)
            
            center_x = x + w // 2
            bottom_y = y + h
            line_end_y = min(bottom_y + 50, frame.shape[0] - 20)
            cv2.line(frame, (center_x, bottom_y), 
                    (center_x, line_end_y), color, 2)
            
            cv2.circle(frame, (center_x, line_end_y), 5, color, -1)
        
        return frame
    
    def _draw_info_panel(self, frame, status):
        h, w = frame.shape[:2]
        panel_width = 250
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (w - panel_width, 0), (w, h), 
                     (100, 100, 100), 2)
        
        x_pos = w - panel_width + 10
        y_pos = 30
        
        cv2.putText(frame, "SYSTEM STATUS", (x_pos, y_pos), 
                   self.font, 0.8, config.COLORS["text"], 2)
        y_pos += 30
        
        cv2.line(frame, (x_pos, y_pos), (w - 10, y_pos), 
                (100, 100, 100), 1)
        y_pos += 20
        
        fps = status.get('fps', 0)
        fps_color = config.COLORS["safe"] if fps > 20 else config.COLORS["caution"]
        cv2.putText(frame, f"FPS: {fps:.1f}", (x_pos, y_pos), 
                   self.font, 0.7, fps_color, 2)
        y_pos += 30
        
        mode = status.get('mode', 'NORMAL')
        cv2.putText(frame, f"Mode: {mode}", (x_pos, y_pos), 
                   self.font, 0.7, config.COLORS["text"], 2)
        y_pos += 30
        
        if status.get('night_mode', False):
            cv2.putText(frame, "NIGHT MODE", (x_pos, y_pos), 
                       self.font, 0.7, (255, 200, 100), 2)
            y_pos += 30
        
        obj_count = status.get('object_count', 0)
        cv2.putText(frame, f"Objects: {obj_count}", (x_pos, y_pos), 
                   self.font, 0.7, config.COLORS["text"], 2)
        y_pos += 30
        
        y_pos += 10
        cv2.line(frame, (x_pos, y_pos), (w - 10, y_pos), 
                (100, 100, 100), 1)
        y_pos += 30
        
        cv2.putText(frame, "CONTROLS:", (x_pos, y_pos), 
                   self.font, 0.7, config.COLORS["text"], 2)
        y_pos += 25
        
        controls = [
            "Q - Quit",
            "N - Night Mode",
            "C - Calibrate",
            "S - Screenshot",
            "M - Mute Audio"
        ]
        
        for control in controls:
            cv2.putText(frame, control, (x_pos, y_pos), 
                       self.font, 0.6, (200, 200, 200), 1)
            y_pos += 22
        
        return frame
    
    def _draw_distance_bars(self, frame, detections, distances):
        h, w = frame.shape[:2]
        bar_height = 100
        margin = 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, h - bar_height - 20), 
                     (w - margin, h - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        bar_width = w - 2 * margin
        max_distance = 5.0
        
        for i in range(6):
            x = margin + int((i / 5) * bar_width)
            dist_label = f"{i}m"
            cv2.line(frame, (x, h - bar_height - 20), 
                    (x, h - 20), (100, 100, 100), 1)
            cv2.putText(frame, dist_label, (x - 10, h - 5), 
                       self.font, 0.5, (200, 200, 200), 1)
        
        if distances:
            min_dist = min(distances.values())
            
            bar_x = margin + int((min_dist / max_distance) * bar_width)
            bar_x = min(bar_x, w - margin)
            
            _, color = self._get_distance_info(min_dist)
            
            triangle_pts = np.array([
                [bar_x, h - bar_height - 30],
                [bar_x - 10, h - bar_height - 10],
                [bar_x + 10, h - bar_height - 10]
            ], np.int32)
            cv2.fillPoly(frame, [triangle_pts], color)
            cv2.polylines(frame, [triangle_pts], True, (255, 255, 255), 2)
            
            dist_text = f"{min_dist:.2f}m"
            cv2.putText(frame, dist_text, (bar_x - 30, h - bar_height - 40), 
                       self.font, 0.7, color, 2)
        
        return frame
    
    def _draw_danger_warning(self, frame, distance):
        h, w = frame.shape[:2]
        self.pulse_phase = (self.pulse_phase + 0.2) % (2 * np.pi)
        alpha = 0.3 + 0.2 * np.sin(self.pulse_phase)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        warning_text = "STOP!"
        (text_w, text_h), _ = cv2.getTextSize(warning_text, self.font, 3, 5)
        text_x = (w - text_w) // 2
        text_y = (h + text_h) // 2
        
        cv2.putText(frame, warning_text, (text_x, text_y), 
                   self.font, 3, (0, 0, 255), 5)
        
        dist_text = f"Object at {distance:.2f}m"
        (text_w2, _), _ = cv2.getTextSize(dist_text, self.font, 1.5, 3)
        cv2.putText(frame, dist_text, ((w - text_w2) // 2, text_y + 60), 
                   self.font, 1.5, (255, 255, 255), 3)
        
        border_thickness = 8 + int(4 * np.sin(self.pulse_phase))
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
        
        return frame
    
    def _draw_caution_warning(self, frame, distance):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 6)
        
        banner_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 200, 200), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        warning_text = f"CAUTION: Object at {distance:.2f}m"
        (text_w, text_h), _ = cv2.getTextSize(warning_text, self.font, 1.2, 3)
        text_x = (w - text_w) // 2
        text_y = (banner_height + text_h) // 2
        
        cv2.putText(frame, warning_text, (text_x, text_y), 
                   self.font, 1.2, (0, 0, 0), 3)
        
        return frame
    
    def _draw_header(self, frame):
        h, w = frame.shape[:2]
        header_height = 40
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        title = "REVERSE PARKING ASSISTANT"
        cv2.putText(frame, title, (20, 30), self.font, 0.8, 
                   (255, 255, 255), 2)
        
        return frame
    
    def _get_distance_info(self, distance):
        if distance >= config.SAFE_DISTANCE:
            return "SAFE", config.COLORS["safe"]
        elif distance >= config.CAUTION_DISTANCE:
            return "CAUTION", config.COLORS["caution"]
        else:
            return "DANGER", config.COLORS["danger"]