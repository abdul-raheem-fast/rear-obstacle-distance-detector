import cv2
import numpy as np
import config


class NightModeEnhancer:
    
    def __init__(self):
        self.enabled = False
        self.brightness = config.NIGHT_MODE_BRIGHTNESS
        self.contrast = config.NIGHT_MODE_CONTRAST
        self.gamma = config.NIGHT_MODE_GAMMA
        
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # Keep realtime stream smooth by default.
        self.noise_reduction = False
    
    def enhance(self, frame):
        if not self.enabled:
            return frame
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        enhanced = self._apply_gamma(enhanced, self.gamma)
        enhanced = cv2.convertScaleAbs(enhanced, 
                                     alpha=self.contrast, 
                                     beta=(self.brightness - 1) * 50)
        
        if self.noise_reduction:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        enhanced = self._enhance_edges(enhanced)
        return enhanced
    
    def _apply_gamma(self, image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _enhance_edges(self, image):
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
    
    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def auto_detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        threshold = 80
        should_enable = mean_brightness < threshold
        return should_enable, mean_brightness


class AdaptiveEnhancer(NightModeEnhancer):
    
    def __init__(self):
        super().__init__()
        self.auto_mode = True
        self.brightness_history = []
        self.history_size = 30
    
    def enhance_adaptive(self, frame):
        should_enable_night, brightness = self.auto_detect(frame)
        
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > self.history_size:
            self.brightness_history.pop(0)
        
        if len(self.brightness_history) > 5:
            recent_avg = np.mean(self.brightness_history[-5:])
            
            if self.auto_mode and recent_avg < 60:
                self.enabled = True
                darkness_factor = 1 - (recent_avg / 100)
                self.gamma = max(0.5, 1.0 - (darkness_factor * 0.3))
                self.contrast = min(2.0, 1.0 + darkness_factor)
            elif self.auto_mode and recent_avg > 100:
                self.enabled = False
        
        return self.enhance(frame)
    
    def get_lighting_report(self):
        if not self.brightness_history:
            return "Unknown"
        
        avg_brightness = np.mean(self.brightness_history)
        
        if avg_brightness < 50:
            return "Very Dark"
        elif avg_brightness < 80:
            return "Dim"
        elif avg_brightness < 120:
            return "Moderate"
        else:
            return "Bright"