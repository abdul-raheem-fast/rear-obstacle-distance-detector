import cv2
import numpy as np
import time
import argparse
import os
from collections import deque

import config
from object_detector import ObjectDetector, YOLODetector
from distance_estimator import DistanceEstimator, AdvancedDistanceEstimator
from gui_display import ParkingGUIDisplay
from audio_manager import AudioManager
from night_mode import NightModeEnhancer, AdaptiveEnhancer
from calibration import Calibrator, MultiPointCalibrator


def _configure_camera_capture(cap):
    # Try to enable auto exposure on Windows webcams to avoid dark frames.
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_GAIN, 0)
    except Exception:
        pass


def _auto_brighten_if_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    if mean_luma >= 55.0:
        return frame
    gain = min(2.2, max(1.1, 80.0 / max(mean_luma, 1.0)))
    return cv2.convertScaleAbs(frame, alpha=gain, beta=0)


class ReverseParkingAssistant:
    
    def __init__(self, camera_index=config.CAMERA_INDEX, use_yolo=False):
        print("=" * 60)
        print("REVERSE PARKING ASSISTANT")
        print("Real-time Distance Detection & Warning System")
        print("=" * 60)
        
        self.camera_index = camera_index
        self.cap = None
        
        print("Initializing modules...")
        if use_yolo:
            try:
                self.detector = YOLODetector()
                print("Using YOLO detector")
            except:
                print("Falling back to MobileNet SSD")
                self.detector = ObjectDetector()
        else:
            self.detector = ObjectDetector()
            print("Using MobileNet SSD detector")
        
        self.estimator = AdvancedDistanceEstimator()
        self.display = ParkingGUIDisplay()
        self.audio = AudioManager()
        self.night_mode = AdaptiveEnhancer()
        self.calibrator = Calibrator()
        
        self.running = False
        self.paused = False
        self.show_help = False
        self.frame_count = 0
        self.fps = 0
        self.fps_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=5)
        self.screenshot_count = 0
        
        print("Initialization complete")
        print("Press 'H' for help, 'Q' to quit")
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_index}")
            print("Available cameras:")
            for i in range(5):
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    print(f"  Camera {i}: Available")
                    test_cap.release()
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        _configure_camera_capture(self.cap)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    
    def process_frame(self, frame):
        if self.night_mode.enabled:
            frame = self.night_mode.enhance_adaptive(frame)
        else:
            frame = _auto_brighten_if_dark(frame)
        
        detections = self.detector.detect(frame)
        self.detection_history.append(detections)
        
        if len(self.detection_history) > 0:
            current_detections = self.detection_history[-1]
        else:
            current_detections = detections
        
        distances = {}
        min_distance = float('inf')
        closest_object = None
        
        for det in current_detections:
            obj_id = f"{det['class']}_{det['bbox'][0]}_{det['bbox'][1]}"
            distance = self.estimator.estimate_distance_with_ground_plane(
                det['bbox'], frame.shape, det['class']
            )
            distances[obj_id] = distance
            
            if distance < min_distance:
                min_distance = distance
                closest_object = det
        
        if min_distance < config.DANGER_DISTANCE:
            self.audio.play_warning('danger', min_distance)
        elif min_distance < config.CAUTION_DISTANCE:
            self.audio.play_warning('caution', min_distance)
        elif min_distance < config.SAFE_DISTANCE:
            self.audio.play_warning('safe', min_distance)
        
        status = {
            'fps': self.fps,
            'mode': 'NIGHT' if self.night_mode.enabled else 'NORMAL',
            'night_mode': self.night_mode.enabled,
            'object_count': len(current_detections),
            'lighting': self.night_mode.get_lighting_report() if self.night_mode.enabled else 'N/A'
        }
        
        if self.calibrator.calibrating:
            display = self.calibrator.process_frame(frame, current_detections)
        else:
            display = self.display.draw_interface(
                frame, current_detections, distances, status
            )
        
        if self.show_help:
            display = self._draw_help_overlay(display)
        
        return display
    
    def _draw_help_overlay(self, frame):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        help_text = [
            "REVERSE PARKING ASSISTANT - HELP",
            "",
            "CONTROLS:",
            "  Q - Quit application",
            "  P - Pause/Resume",
            "  N - Toggle Night Mode",
            "  A - Toggle Auto Night Mode",
            "  C - Start Calibration",
            "  SPACE - Capture Calibration",
            "  ESC - Cancel Calibration",
            "  M - Toggle Mute",
            "  S - Take Screenshot",
            "  H - Toggle this Help",
            "",
            "DISTANCE ZONES:",
            f"  Green (>{config.SAFE_DISTANCE}m) - Safe",
            f"  Yellow ({config.CAUTION_DISTANCE}-{config.SAFE_DISTANCE}m) - Caution",
            f"  Red (<{config.CAUTION_DISTANCE}m) - Danger",
            "",
            "Press H to close this help"
        ]
        
        y_pos = 50
        for line in help_text:
            cv2.putText(frame, line, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        return frame
    
    def handle_key(self, key):
        if key == ord('q') or key == 27:
            return False
        
        elif key == ord('p'):
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        
        elif key == ord('n'):
            state = self.night_mode.toggle()
            print(f"Night mode: {'ON' if state else 'OFF'}")
        
        elif key == ord('a'):
            self.night_mode.auto_mode = not self.night_mode.auto_mode
            print(f"Auto night mode: {'ON' if self.night_mode.auto_mode else 'OFF'}")
        
        elif key == ord('c'):
            self.calibrator.start_calibration(reference_distance=2.0, reference_width=0.5)
        
        elif key == ord(' ') and self.calibrator.calibrating:
            if len(self.detection_history) > 0 and len(self.detection_history[-1]) > 0:
                success, focal = self.calibrator.capture_calibration(0, self.detection_history[-1])
                if success:
                    self.estimator.focal_length = focal
                    print(f"Calibration complete. Focal length: {focal:.2f}")
                    self.calibrator.finish_calibration()
        
        elif key == 27 and self.calibrator.calibrating:
            self.calibrator.cancel_calibration()
        
        elif key == ord('m'):
            state = self.audio.toggle_mute()
            print(f"Audio: {'MUTED' if state else 'ON'}")
        
        elif key == ord('s'):
            self._take_screenshot()
        
        elif key == ord('h'):
            self.show_help = not self.show_help
        
        return True
    
    def _take_screenshot(self):
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:03d}.png"
        
        if hasattr(self, 'last_display'):
            cv2.imwrite(filename, self.last_display)
            print(f"Screenshot saved: {filename}")
    
    def _calculate_fps(self):
        self.frame_count += 1
        current_time = time.time()
        
        if not hasattr(self, 'fps_start_time'):
            self.fps_start_time = current_time
        
        elapsed = current_time - self.fps_start_time
        
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.fps_history.append(self.fps)
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def run(self):
        if not self.start_camera():
            return
        
        self.running = True
        print("Starting main loop...")
        print("Press 'Q' to quit, 'H' for help")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Frame capture failed")
                    time.sleep(0.1)
                    continue
                
                self._calculate_fps()
                
                if not self.paused:
                    display = self.process_frame(frame)
                else:
                    display = getattr(self, 'last_display', frame)
                
                self.last_display = display
                
                cv2.imshow("Reverse Parking Assistant", display)
                
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("Shutting down...")
        
        self.running = False
        self.audio.stop_current()
        self.audio.cleanup()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("Shutdown complete")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Reverse Parking Assistant - Real-time Distance Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default webcam
  python main.py --camera 1         # Use external camera
  python main.py --yolo             # Use YOLO detector (slower, more accurate)
  python main.py --night            # Start with night mode enabled
        """
    )
    
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--yolo", action="store_true",
                       help="Use YOLO detector instead of MobileNet")
    parser.add_argument("--night", action="store_true",
                       help="Start with night mode enabled")
    parser.add_argument("--width", type=int, default=1280,
                       help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="Camera height (default: 720)")
    
    args = parser.parse_args()
    
    config.CAMERA_INDEX = args.camera
    config.FRAME_WIDTH = args.width
    config.FRAME_HEIGHT = args.height
    
    app = ReverseParkingAssistant(
        camera_index=args.camera,
        use_yolo=args.yolo
    )
    
    if args.night:
        app.night_mode.enabled = True
    
    app.run()


if __name__ == "__main__":
    main()
