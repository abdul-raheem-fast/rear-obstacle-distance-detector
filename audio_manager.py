import pygame
import threading
import time
import config
import os
import numpy as np


class AudioManager:
    def __init__(self, enabled=config.ENABLE_AUDIO_WARNINGS):
        self.enabled = enabled
        self.muted = False
        self.last_warning_time = {
            'danger': 0,
            'caution': 0,
            'safe': 0
        }
        self.cooldown = config.WARNING_COOLDOWN
        
        if self.enabled:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                print("[INFO] Audio system initialized")
            except Exception as e:
                print(f"[WARNING] Audio initialization failed: {e}")
                self.enabled = False
        
        self.sounds_dir = "sounds"
        if not os.path.exists(self.sounds_dir):
            os.makedirs(self.sounds_dir)
        
        self.sounds = self._load_sounds()
        self.current_thread = None
        self.stop_event = threading.Event()
    
    def _load_sounds(self):
        sounds = {}
        
        sound_files = {
            'danger': 'danger.wav',
            'caution': 'caution.wav',
            'safe': 'safe.wav'
        }
        
        for level, filename in sound_files.items():
            filepath = os.path.join(self.sounds_dir, filename)
            if os.path.exists(filepath):
                try:
                    sounds[level] = pygame.mixer.Sound(filepath)
                except:
                    sounds[level] = None
            else:
                sounds[level] = None
        
        return sounds
    
    def generate_beep(self, frequency, duration, volume=0.5):
        sample_rate = 44100
        num_samples = int(duration * sample_rate)
        
        t = np.linspace(0, duration, num_samples, False)
        wave = np.sin(frequency * t * 2 * np.pi)
        
        fade_samples = int(0.01 * sample_rate)
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        audio = (wave * volume * 32767).astype(np.int16)
        stereo_audio = np.column_stack((audio, audio))
        
        return stereo_audio
    
    def play_warning(self, level, distance=None):
        if not self.enabled or self.muted:
            return
        
        current_time = time.time()
        
        if current_time - self.last_warning_time[level] < self.cooldown:
            return
        
        self.last_warning_time[level] = current_time
        self.stop_current()
        
        if level == 'danger':
            self._play_danger_warning(distance)
        elif level == 'caution':
            self._play_caution_warning(distance)
        elif level == 'safe':
            self._play_safe_notification()
    
    def _play_danger_warning(self, distance):
        def danger_loop():
            while not self.stop_event.is_set():
                if self.sounds.get('danger'):
                    self.sounds['danger'].play()
                else:
                    self._generate_and_play_beep(1000, 0.15)
                
                time.sleep(0.2)
                
                if self.stop_event.is_set():
                    break
        
        self.stop_event.clear()
        self.current_thread = threading.Thread(target=danger_loop)
        self.current_thread.daemon = True
        self.current_thread.start()
    
    def _play_caution_warning(self, distance):
        def caution_loop():
            beep_count = 0
            while not self.stop_event.is_set() and beep_count < 3:
                if self.sounds.get('caution'):
                    self.sounds['caution'].play()
                else:
                    self._generate_and_play_beep(800, 0.3)
                
                time.sleep(0.6)
                beep_count += 1
        
        self.stop_event.clear()
        self.current_thread = threading.Thread(target=caution_loop)
        self.current_thread.daemon = True
        self.current_thread.start()
    
    def _play_safe_notification(self):
        if self.sounds.get('safe'):
            self.sounds['safe'].play()
        else:
            self._generate_and_play_beep(500, 0.5)
    
    def _generate_and_play_beep(self, frequency, duration):
        try:
            import numpy as np
            audio = self.generate_beep(frequency, duration)
            sound = pygame.sndarray.make_sound(audio)
            sound.play()
        except Exception as e:
            print(f"[WARNING] Could not generate beep: {e}")
    
    def stop_current(self):
        self.stop_event.set()
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=0.1)
        pygame.mixer.stop()
    
    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.stop_current()
        return self.muted
    
    def cleanup(self):
        self.stop_current()
        if self.enabled:
            pygame.mixer.quit()


class VoiceWarningSystem(AudioManager):
    
    def __init__(self, enabled=config.ENABLE_AUDIO_WARNINGS):
        super().__init__(enabled)
        self.voice_enabled = True
        self.last_announcement = ""
        self.announcement_cooldown = 3.0
    
    def announce_distance(self, distance):
        if not self.enabled or self.muted:
            return
        
        current_time = time.time()
        
        if distance < config.DANGER_DISTANCE:
            announcement = "danger"
            pitch = 1000
        elif distance < config.CAUTION_DISTANCE:
            announcement = "caution"
            pitch = 700
        else:
            announcement = "safe"
            pitch = 500
        
        if (announcement != self.last_announcement and 
            current_time - self.last_warning_time.get(announcement, 0) > self.announcement_cooldown):
            
            self.last_announcement = announcement
            self.last_warning_time[announcement] = current_time
            self._play_distance_tone(pitch, distance)
    
    def _play_distance_tone(self, base_pitch, distance):
        urgency_factor = max(0, 1 - (distance / config.SAFE_DISTANCE))
        final_pitch = int(base_pitch + (urgency_factor * 500))
        duration = 0.3 + (urgency_factor * 0.4)
        self._generate_and_play_beep(final_pitch, duration)