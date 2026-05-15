"""
Web Server for Reverse Parking Assistant
Streams processed video and detection data to mobile browser GUI
"""

import cv2
import numpy as np
import time
import json
import threading
import socket
from collections import deque
from flask import Flask, Response, render_template, jsonify, request

import config
from object_detector import ObjectDetector, YOLODetector
from distance_estimator import DistanceEstimator, AdvancedDistanceEstimator
from night_mode import NightModeEnhancer, AdaptiveEnhancer

app = Flask(__name__, template_folder='templates', static_folder='static')

import base64

# Global State

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.night_mode_enabled = False
        self.auto_night_mode = False
        self.muted = False
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_history = deque(maxlen=30)

        # Latest detection results
        self.latest_detections = []
        self.latest_distances = {}
        self.min_distance = float('inf')
        self.closest_class = ""
        self.object_count = 0
        self.lighting = "N/A"
        self.mode = "NORMAL"
        self.warning_level = "safe"  # safe / caution / danger

        # CV modules
        self.detector = ObjectDetector()
        self.estimator = AdvancedDistanceEstimator()
        self.night_enhancer = AdaptiveEnhancer()
        self.latest_frame_bytes = None
        self.capture_thread = None
        self.capture_running = False
        self.server_frame_counter = 0
        self.cached_server_detections = []
        self.cached_server_distances = {}
        self.cached_server_min_distance = float('inf')
        self.cached_server_closest_class = ""
        self.cached_server_warning_level = "safe"
        self.detector_paused_until = 0.0
        
        # Camera switching
        self.camera_mode = "laptop"  # "laptop" or "mobile"
        self.laptop_cap = None
        self.mobile_cap = None
        self.current_cap = None

state = AppState()

# Helper Functions

def get_local_ip():
    """Get machine's local IP so mobile can connect."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def find_available_cameras():
    """Find all available cameras and their indices."""
    available = []
    for i in range(5):  # Check indices 0-4
        cap = _open_camera_index(i)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def calculate_fps():
    state.frame_count += 1
    elapsed = time.time() - state.fps_start_time
    if elapsed > 1.0:
        state.fps = state.frame_count / elapsed
        state.fps_history.append(state.fps)
        state.frame_count = 0
        state.fps_start_time = time.time()


def _configure_camera_capture(cap):
    # Try to enable auto exposure on Windows webcams to avoid black frames.
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
    
    # Set camera properties
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass


def _open_camera_index(index):
    try:
        # On Windows, CAP_DSHOW often works best with some webcams
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    except Exception:
        pass
        
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
    except Exception:
        pass
        
    return None


def _stop_capture_loop():
    state.capture_running = False
    if state.capture_thread and state.capture_thread.is_alive():
        state.capture_thread = None
    if state.current_cap is not None:
        state.current_cap.release()
    if state.laptop_cap is not None:
        state.laptop_cap.release()
    if state.mobile_cap is not None:
        state.mobile_cap.release()
    state.current_cap = None
    state.laptop_cap = None
    state.mobile_cap = None
    state.latest_frame_bytes = None


def draw_overlay(frame, detections, distances, min_dist, warning_level):
    """Draw bounding boxes, labels, distance bars on frame for the video stream."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Parking guide lines
    cx = w // 2
    pts_l = np.array([[cx - 40, h], [cx - 120, h - 80], [cx - 160, h - 160]], np.int32)
    pts_r = np.array([[cx + 40, h], [cx + 120, h - 80], [cx + 160, h - 160]], np.int32)
    # Draw green parking guide lines with thicker width
    cv2.polylines(display, [pts_l], False, (0, 255, 0), 3)
    cv2.polylines(display, [pts_r], False, (0, 255, 0), 3)

    # Detection boxes
    for det in detections:
        x, y, bw, bh = det['bbox']
        cls = det['class']
        conf = det['confidence']
        obj_id = f"{cls}_{x}_{y}"
        dist = distances.get(obj_id, float('inf'))

        if dist < config.DANGER_DISTANCE:
            color = (0, 0, 255)
        elif dist < config.CAUTION_DISTANCE:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        thickness = 3 if dist < config.DANGER_DISTANCE else 2
        cv2.rectangle(display, (x, y), (x + bw, y + bh), color, thickness)

        label = f"{cls.upper()}: {dist:.2f}m ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(display, (x, y - th - 8), (x + tw + 8, y), color, -1)
        cv2.putText(display, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Danger overlay
    if warning_level == "danger":
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        txt = "STOP!"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
        cv2.putText(display, txt, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
    elif warning_level == "caution":
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 255, 255), 4)

    return display


def _serialize_bbox(bbox):
    return [int(v) for v in bbox]


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _auto_brighten_if_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    if mean_luma >= 55.0:
        return frame
    # Lift exposure in a controlled way when camera auto exposure fails.
    gain = min(2.2, max(1.1, 80.0 / max(mean_luma, 1.0)))
    return cv2.convertScaleAbs(frame, alpha=gain, beta=0)


def switch_camera(mode):
    """Switch camera mode between laptop and mobile"""
    with state.lock:
        if mode == state.camera_mode and state.current_cap is not None and state.current_cap.isOpened():
            return True
        
        # Stop current camera
        if state.current_cap:
            state.current_cap.release()
            state.current_cap = None
        
        # Start new camera
        if mode == "laptop":
            preferred_indices = [0, 1, 2, 3, 4]
            state.laptop_cap = None
            selected_index = None
            
            for camera_index in preferred_indices:
                cap = _open_camera_index(camera_index)
                if cap is not None and cap.isOpened():
                    state.laptop_cap = cap
                    selected_index = camera_index
                    print(f"Using camera index {camera_index} (laptop mode)")
                    break

            if state.laptop_cap is None or not state.laptop_cap.isOpened():
                print("[ERROR] Failed to open laptop camera")
                return False
            
            _configure_camera_capture(state.laptop_cap)
            state.current_cap = state.laptop_cap
            state.camera_mode = "laptop"
            print(f"Switched to LAPTOP camera (index {selected_index})")
            return True
            
        elif mode == "mobile":
            state.camera_mode = "mobile"
            state.latest_frame_bytes = None
            print("Switched to MOBILE camera (phone capture)")
            return True
        
        return False


# Video Capture Loop

def _process_frame_for_state(frame):
    calculate_fps()

    if state.night_enhancer.enabled or state.auto_night_mode:
        state.night_enhancer.auto_mode = state.auto_night_mode
        frame = state.night_enhancer.enhance_adaptive(frame)
        state.lighting = state.night_enhancer.get_lighting_report()
        state.mode = "NIGHT" if state.night_enhancer.enabled else "NORMAL"
    else:
        frame = _auto_brighten_if_dark(frame)
        state.lighting = "N/A"
        state.mode = "NORMAL"

    state.server_frame_counter += 1
    run_detection_now = (state.server_frame_counter % 3 == 0)

    if run_detection_now:
        now = time.time()
        if now < state.detector_paused_until:
            detections = []
            print("[DEBUG] Detector paused")
        else:
            detect_start = time.time()
            print(f"[DEBUG] Running detection on frame {state.server_frame_counter}")
            detections = state.detector.detect(frame)
            detect_ms = (time.time() - detect_start) * 1000.0
            print(f"[DEBUG] Detection took {detect_ms:.1f}ms, found {len(detections)} objects")
            # Fail-safe: if detection gets too slow, pause briefly to keep stream alive.
            if detect_ms > 800:
                state.detector_paused_until = time.time() + 2.0
                print("[DEBUG] Detector paused for 2s (too slow)")
        distances = {}
        min_distance = float('inf')
        closest_cls = ""

        for det in detections:
            obj_id = f"{det['class']}_{det['bbox'][0]}_{det['bbox'][1]}"
            d = state.estimator.estimate_distance_with_ground_plane(
                det['bbox'], frame.shape, det['class']
            )
            distances[obj_id] = d
            if d < min_distance:
                min_distance = d
                closest_cls = det['class']

        if min_distance < config.DANGER_DISTANCE:
            wl = "danger"
        elif min_distance < config.CAUTION_DISTANCE:
            wl = "caution"
        elif min_distance < config.SAFE_DISTANCE:
            wl = "warning"
        else:
            wl = "safe"

        state.cached_server_detections = detections
        state.cached_server_distances = distances
        state.cached_server_min_distance = min_distance
        state.cached_server_closest_class = closest_cls
        state.cached_server_warning_level = wl
    else:
        detections = state.cached_server_detections
        distances = state.cached_server_distances
        min_distance = state.cached_server_min_distance
        closest_cls = state.cached_server_closest_class
        wl = state.cached_server_warning_level

    with state.lock:
        state.latest_detections = detections
        state.latest_distances = distances
        state.min_distance = min_distance
        state.closest_class = closest_cls
        state.object_count = len(detections)
        state.warning_level = wl

    return draw_overlay(frame, detections, distances, min_distance, wl)


def capture_loop():
    failed_reads = 0
    print("[INFO] Background camera loop started")
    
    # Show available cameras
    available = find_available_cameras()
    print(f"[INFO] Available cameras: {available}")
    
    # Initialize with laptop camera by default
    if not switch_camera("laptop"):
        print("[ERROR] Failed to initialize any camera")
        return

    consecutive_failures = 0
    
    while state.capture_running:
        # CRITICAL: Only process laptop camera if in laptop mode
        if state.camera_mode == "mobile":
            if state.current_cap is not None:
                state.current_cap.release()
                state.current_cap = None
            # Clear laptop detections when in mobile mode
            with state.lock:
                if state.latest_detections and len(state.latest_detections) > 0:
                    print("[INFO] Clearing laptop detections (mobile mode active)")
                    state.latest_detections = []
                    state.latest_distances = {}
                    state.min_distance = float('inf')
                    state.closest_class = ""
                    state.object_count = 0
                    state.warning_level = "safe"
            time.sleep(0.2)
            continue
            
        if state.current_cap is None or not state.current_cap.isOpened():
            consecutive_failures += 1
            if consecutive_failures <= 3:
                print(f"[WARN] Camera disconnected, reconnecting (attempt {consecutive_failures}/3)")
                time.sleep(1.0)
                if switch_camera(state.camera_mode):
                    consecutive_failures = 0
                    print(f"[INFO] Successfully reconnected camera")
            else:
                print(f"[ERROR] Camera reconnection failed, waiting...")
                time.sleep(3.0)
            continue

        ret, frame = state.current_cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print(f"[WARN] Multiple read failures, reconnecting camera")
                if state.current_cap:
                    state.current_cap.release()
                    state.current_cap = None
                time.sleep(0.5)
            else:
                time.sleep(0.05)
            continue
        
        consecutive_failures = 0

        failed_reads = 0
        try:
            display = _process_frame_for_state(frame)
            ok, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with state.lock:
                    state.latest_frame_bytes = buffer.tobytes()
        except Exception as err:
            print(f"[WARN] Frame process error: {err}")
            time.sleep(0.03)

    if state.current_cap is not None:
        state.current_cap.release()
    print("[INFO] Background camera loop stopped")


def ensure_capture_running():
    with state.capture_lock:
        if state.capture_thread and state.capture_thread.is_alive():
            return
        state.capture_running = True
        state.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        state.capture_thread.start()


def generate_frames():
    """Yield latest processed bytes from shared camera loop."""
    if state.camera_mode == "laptop":
        ensure_capture_running()
    while True:
        if state.camera_mode == "mobile":
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "WORKING ON PHONE", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ok, buffer = cv2.imencode('.jpg', err_frame)
            if ok:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.2)
            continue
        with state.lock:
            frame_bytes = state.latest_frame_bytes

        if frame_bytes is None:
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "WAITING FOR SERVER CAMERA...", (95, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ok, buffer = cv2.imencode('.jpg', err_frame)
            if ok:
                frame_bytes = buffer.tobytes()
            time.sleep(0.05)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Only run the generator if requested
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """Returns JSON with current detection data for the GUI to poll."""
    with state.lock:
        # CRITICAL: Only return detections if in laptop mode
        if state.camera_mode != 'laptop':
            # Return empty detections when not in laptop mode
            return jsonify({
                'fps': 0,
                'mode': state.mode,
                'lighting': 'N/A',
                'object_count': 0,
                'min_distance': None,
                'closest_class': '',
                'warning_level': 'safe',
                'night_mode': state.night_enhancer.enabled,
                'auto_night_mode': state.auto_night_mode,
                'muted': state.muted,
                'detections': [],
                'safe_distance': _safe_float(config.SAFE_DISTANCE),
                'caution_distance': _safe_float(config.CAUTION_DISTANCE),
                'danger_distance': _safe_float(config.DANGER_DISTANCE),
                'camera_mode': state.camera_mode,
            })
        
        det_list = []
        for det in state.latest_detections:
            obj_id = f"{det['class']}_{det['bbox'][0]}_{det['bbox'][1]}"
            det_list.append({
                'class': det['class'],
                'confidence': round(float(det['confidence']), 2),
                'distance': round(float(state.latest_distances.get(obj_id, 999)), 2),
                'bbox': _serialize_bbox(det['bbox'])
            })

        return jsonify({
            'fps': round(float(state.fps), 1),
            'mode': state.mode,
            'lighting': state.lighting,
            'object_count': int(state.object_count),
            'min_distance': round(float(state.min_distance), 2) if state.min_distance != float('inf') else None,
            'closest_class': state.closest_class,
            'warning_level': state.warning_level,
            'night_mode': state.night_enhancer.enabled,
            'auto_night_mode': state.auto_night_mode,
            'muted': state.muted,
            'detections': det_list,
            'safe_distance': _safe_float(config.SAFE_DISTANCE),
            'caution_distance': _safe_float(config.CAUTION_DISTANCE),
            'danger_distance': _safe_float(config.DANGER_DISTANCE),
            'camera_mode': state.camera_mode,
        })


@app.route('/api/switch_camera', methods=['POST'])
def switch_camera_api():
    """Switch camera mode between laptop and mobile"""
    data = request.get_json(force=True)
    mode = data.get('mode', 'laptop')
    
    if mode not in ['laptop', 'mobile']:
        return jsonify({'error': 'Invalid camera mode'}), 400
    
    success = switch_camera(mode)
    return jsonify({
        'success': success,
        'camera_mode': state.camera_mode
    })


@app.route('/api/process', methods=['POST'])
def process_frame():
    # CRITICAL: Only process if in mobile mode
    if state.camera_mode != 'mobile':
        return jsonify({'error': 'Not in mobile camera mode'}), 400
    
    try:
        data = request.get_json(force=True)
        if 'image' not in data:
            return jsonify({'error': 'no image'}), 400

        # Decode base64 image from phone camera
        image_data = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'invalid image'}), 400

        # Resize frame for faster processing on phone
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w = 640
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        calculate_fps()

        # Night-mode enhancement (lightweight)
        if state.night_enhancer.enabled or state.auto_night_mode:
            state.night_enhancer.auto_mode = state.auto_night_mode
            frame = state.night_enhancer.enhance_adaptive(frame)
            state.lighting = state.night_enhancer.get_lighting_report()
            state.mode = "NIGHT" if state.night_enhancer.enabled else "NORMAL"
        else:
            state.lighting = "N/A"
            state.mode = "NORMAL"

        # Object detection with aggressive throttling for mobile
        now = time.time()
        if now < state.detector_paused_until:
            # Use cached detections
            detections = state.cached_server_detections
            distances = state.cached_server_distances
            min_distance = state.cached_server_min_distance
            closest_cls = state.cached_server_closest_class
            wl = state.cached_server_warning_level
        else:
            detect_start = time.time()
            detections = state.detector.detect(frame)
            detect_ms = (time.time() - detect_start) * 1000.0
            
            # More aggressive throttling for mobile
            if detect_ms > 500:
                state.detector_paused_until = time.time() + 1.5
                print(f"[MOBILE] Detection slow ({detect_ms:.0f}ms), throttling")

            # Distance estimation
            distances = {}
            min_distance = float('inf')
            closest_cls = ""
            for det in detections:
                obj_id = f"{det['class']}_{det['bbox'][0]}_{det['bbox'][1]}"
                d = state.estimator.estimate_distance_with_ground_plane(
                    det['bbox'], frame.shape, det['class']
                )
                distances[obj_id] = d
                if d < min_distance:
                    min_distance = d
                    closest_cls = det['class']

            # Warning level
            if min_distance < config.DANGER_DISTANCE:
                wl = "danger"
            elif min_distance < config.CAUTION_DISTANCE:
                wl = "caution"
            elif min_distance < config.SAFE_DISTANCE:
                wl = "warning"
            else:
                wl = "safe"

            # Cache results
            state.cached_server_detections = detections
            state.cached_server_distances = distances
            state.cached_server_min_distance = min_distance
            state.cached_server_closest_class = closest_cls
            state.cached_server_warning_level = wl

        # CRITICAL: Update shared state ONLY for mobile mode
        # Do NOT update state.latest_detections to prevent laptop from seeing mobile detections
        # with state.lock:
        #     state.latest_detections = detections
        #     state.latest_distances = distances
        #     state.min_distance = min_distance
        #     state.closest_class = closest_cls
        #     state.object_count = len(detections)
        #     state.warning_level = wl

        # Draw overlay on frame
        display = draw_overlay(frame, detections, distances, min_distance, wl)

        # Encode as JPEG with lower quality for faster transmission
        ret2, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 50])
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        det_list = []
        for det in detections:
            obj_id = f"{det['class']}_{det['bbox'][0]}_{det['bbox'][1]}"
            det_list.append({
                'class': det['class'],
                'confidence': round(float(det['confidence']), 2),
                'distance': round(float(distances.get(obj_id, 999)), 2),
                'bbox': _serialize_bbox(det['bbox'])
            })

        return jsonify({
            'image': processed_b64,
            'stats': {
                'fps': round(float(state.fps), 1),
                'mode': state.mode,
                'lighting': state.lighting,
                'object_count': len(detections),
                'min_distance': round(float(min_distance), 2) if min_distance != float('inf') else None,
                'closest_class': closest_cls,
                'warning_level': wl,
                'night_mode': state.night_enhancer.enabled,
                'auto_night_mode': state.auto_night_mode,
                'muted': state.muted,
                'detections': det_list,
                'safe_distance': _safe_float(config.SAFE_DISTANCE),
                'caution_distance': _safe_float(config.CAUTION_DISTANCE),
                'danger_distance': _safe_float(config.DANGER_DISTANCE),
            }
        })
    except Exception as e:
        print(f"[ERROR] /api/process failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    """Toggle settings from the mobile GUI."""
    data = request.get_json(force=True)
    action = data.get('action', '')

    if action == 'night_mode':
        state.night_enhancer.enabled = not state.night_enhancer.enabled
        return jsonify({'night_mode': state.night_enhancer.enabled})

    elif action == 'auto_night':
        state.auto_night_mode = not state.auto_night_mode
        state.night_enhancer.auto_mode = state.auto_night_mode
        return jsonify({'auto_night_mode': state.auto_night_mode})

    elif action == 'mute':
        state.muted = not state.muted
        return jsonify({'muted': state.muted})

    return jsonify({'error': 'unknown action'}), 400


# Entry Point

if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 5000

    print("=" * 60)
    print("  REVERSE PARKING ASSISTANT — Mobile Phone Camera Mode")
    print("=" * 60)
    print(f"  Mobile:  https://{local_ip}:{port}")
    print("=" * 60)
    print("  Open the Mobile URL on your phone's browser")
    print("  (Make sure to accept the self-signed SSL warning!)")
    print("=" * 60)

    # Start camera capture loop
    state.capture_running = True
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, ssl_context='adhoc')
    finally:
        state.capture_running = False
        if capture_thread:
            capture_thread.join(timeout=2)
