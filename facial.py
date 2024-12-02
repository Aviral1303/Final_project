import cv2 
import numpy as np
from deepface import DeepFace
from datetime import datetime
from collections import deque
import threading
import tkinter as tk
from PIL import Image, ImageTk

class EmotionDetector:
    def __init__(self):
        self.emotion_history = deque(maxlen=100)
        self.emotion_timestamps = deque(maxlen=100)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, frame):
        try:
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Use the first detected face
                (x, y, w, h) = faces[0]
                
                # Crop the face region for emotion detection
                face_img = frame[y:y+h, x:x+w]
                
                # Using Deepface for emotion detection
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)[0]
                emotion_label = result.get('dominant_emotion', None)
                return (x, y, w, h), emotion_label, result['emotion'].get(emotion_label, 0)
            
            return None, None, 0
        except Exception as e:
            print(f"Error in detecting emotion: {e}")
            return None, None, 0

    def get_emotion_stats(self):
        if not self.emotion_history:
            return {}
        
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(self.emotion_history)
        return {emotion: count / total * 100 for emotion, count in emotion_counts.items()}

    def reset_emotion_history(self):
        self.emotion_history.clear()
        self.emotion_timestamps.clear()

# facial emotion detection with UI control
class EmotionDetectionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.detector = EmotionDetector()
        self.detection_active = False
        self.root = tk.Tk()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Emotion Detection Control Panel")

        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=10)

        self.camera_frame = tk.Label(self.root)
        self.camera_frame.pack()

        self.emotion_stats_label = tk.Label(self.root, text="Emotion Statistics will be displayed here.")
        self.emotion_stats_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()
        self.root.mainloop()

    def start_detection(self):
        if not self.detection_active:
            self.detector.reset_emotion_history()  # Reset history when starting a new detection session
            self.detection_active = True
            threading.Thread(target=self.run_detection, daemon=True).start()

    def stop_detection(self):
        if self.detection_active:
            self.detection_active = False
            stats = self.detector.get_emotion_stats()
            if stats:
                stats_text = "Emotion statistics:\n" + "\n".join([f"{emotion}: {percentage:.2f}%" for emotion, percentage in stats.items()])
                self.emotion_stats_label.config(text=stats_text)

    def run_detection(self):
        while self.detection_active:
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Flip frame horizontally for natural mirror effect
            # frame = cv2.flip(frame, 1)
            
            # Detect emotion and face from the entire frame
            face_box, emotion, confidence = self.detector.detect_emotion(frame)
            
            if emotion:
                self.detector.emotion_history.append(emotion)
                self.detector.emotion_timestamps.append(datetime.now())

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Detect face and draw bounding box
            face_box, emotion, confidence = self.detector.detect_emotion(frame)
            if face_box:
                (x, y, w, h) = face_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add emotion label
                if emotion:
                    label = f"{emotion}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.detection_active = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    EmotionDetectionApp()
