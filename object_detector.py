import cv2
import numpy as np
import config


class ObjectDetector:
    
    def __init__(self, confidence_threshold=config.CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = config.NMS_THRESHOLD
        
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        
        # Keep all model classes except background so users can see detections
        # in varied scenes (room objects, monitors, etc.).
        self.relevant_classes = None
        
        self.net = self._load_model()
        
    def _load_model(self):
        model_path = "MobileNetSSD_deploy.caffemodel"
        config_path = "MobileNetSSD_deploy.prototxt"
        
        try:
            net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Object detection model loaded successfully")
            return net
        except Exception as e:
            print(f"Could not load model files: {e}")
            print("Running in fallback mode - detecting motion only")
            return None
    
    def detect(self, frame):
        if self.net is None:
            return self._fallback_detection(frame)
        
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                     (300, 300), 127.5)
        
        self.net.setInput(blob)
        try:
            detections = self.net.forward()
        except Exception as e:
            print(f"Detector forward failed, using fallback: {e}")
            return self._fallback_detection(frame)
        
        results = []
        
        print(f"[DETECT] Threshold: {self.confidence_threshold}, Total raw: {detections.shape[2]}")
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            if i < 5:  # Print first 5
                print(f"[DETECT] #{i}: class={class_id}, conf={confidence:.3f}")
            
            if class_id < 0 or class_id >= len(self.classes):
                continue
            class_name = self.classes[class_id]

            if confidence < self.confidence_threshold:
                continue
            
            if class_name != "background" and (
                self.relevant_classes is None or class_name in self.relevant_classes
            ):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, min(startX, w - 1))
                startY = max(0, min(startY, h - 1))
                endX = max(0, min(endX, w - 1))
                endY = max(0, min(endY, h - 1))
                if endX <= startX or endY <= startY:
                    continue
                
                bbox = (startX, startY, endX - startX, endY - startY)
                
                results.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': bbox
                })
        
        print(f"Before NMS: {len(results)} detections")
        results = self._apply_nms(results)
        print(f"After NMS: {len(results)} detections")
        
        return results
    
    def _apply_nms(self, detections):
        if not detections:
            return []
        
        boxes = [d['bbox'] for d in detections]
        scores = [d['confidence'] for d in detections]

        indices = cv2.dnn.NMSBoxes(boxes, scores, 
                                   self.confidence_threshold, 
                                   self.nms_threshold)
        
        if indices is not None and len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[1]
            filtered = [detections[i] for i in indices.flatten()]
            return filtered
        
        return detections
    
    def _fallback_detection(self, frame):
        if not hasattr(self, 'bg_subtractor'):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=50, detectShadows=False)
            self.prev_frame = None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                (x, y, w, h) = cv2.boundingRect(contour)
                detections.append({
                    'class': 'obstacle',
                    'confidence': 0.7,
                    'bbox': (x, y, w, h)
                })
        
        self.prev_frame = gray
        return detections


class YOLODetector(ObjectDetector):
    
    def __init__(self, model_path="yolov4-tiny.weights", 
                 config_path="yolov4-tiny.cfg",
                 confidence_threshold=0.5):
        super().__init__(confidence_threshold)
        self.model_path = model_path
        self.config_path = config_path
        self.net = self._load_yolo_model()
    
    def _load_yolo_model(self):
        try:
            net = cv2.dnn.readNet(self.model_path, self.config_path)
            
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA for YOLO detection")
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.layer_names = net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] 
                                 for i in net.getUnconnectedOutLayers()]
            
            print("YOLO model loaded successfully")
            return net
            
        except Exception as e:
            print(f"Could not load YOLO model: {e}")
            return None
    
    def detect(self, frame):
        if self.net is None:
            return super().detect(frame)
        
        (H, W) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 
                                self.nms_threshold)
        
        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                if class_name in self.relevant_classes or class_name == "unknown":
                    results.append({
                        'class': class_name,
                        'confidence': confidences[i],
                        'bbox': tuple(boxes[i])
                    })
        
        return results