from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QDesktopWidget,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import QTimer, Qt
import cv2
import os
from PyQt5.QtWidgets import QLabel, QPushButton, QApplication, QMainWindow
from PyQt5.QtGui import QIcon, QPixmap
import torch
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

emotion_model = load_model('my_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class AgeDetectionUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Recognition for Justice System")
        self.setStyleSheet("background-color: white;")

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)

        self.move(qtRectangle.topLeft())

        self.setGeometry(300, 300, 900, 800)
        
        uniform_width = 200
        uniform_height = 200
        
        def load_and_resize_image(filepath, width, height):
            pixmap = QPixmap(filepath)
            return pixmap.scaled(width, height)

        select_image_pixmap = load_and_resize_image('images.png', uniform_width, uniform_height)
        self.select_image_button = QPushButton('', self)
        self.select_image_button.setIcon(QIcon(select_image_pixmap))
        self.select_image_button.setIconSize(select_image_pixmap.size())
        self.select_image_button.setGeometry(100, 20, uniform_width, uniform_height)

        select_video_pixmap = load_and_resize_image('camera.png', uniform_width, uniform_height)
        self.select_video_button = QPushButton('', self)
        self.select_video_button.setIcon(QIcon(select_video_pixmap))
        self.select_video_button.setIconSize(select_video_pixmap.size())
        self.select_video_button.setGeometry(350, 20, uniform_width, uniform_height)

        select_camera_pixmap = load_and_resize_image('video.png', uniform_width, uniform_height)
        self.select_camera_button = QPushButton('', self)
        self.select_camera_button.setIcon(QIcon(select_camera_pixmap))
        self.select_camera_button.setIconSize(select_camera_pixmap.size())
        self.select_camera_button.setGeometry(600, 20, uniform_width, uniform_height)
        self.select_image_button.clicked.connect(self.select_image)
        self.select_video_button.clicked.connect(self.select_camera)
        self.select_camera_button.clicked.connect(self.select_video)

        self.image_path = None
        self.video_path = None
        self.video_playing = False
        self.video_frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.faces = None

        self.image_label = QLabel(self)
        self.image_label.setGeometry(100, 300, 600, 400)
        def closeEvent(self, event):
            if self.cap is not None:
                self.cap.release()

    def display_resized_image(self, image, target_width=600, target_height=400):
        if image is not None:
            resized_img = cv2.resize(image, (target_width, target_height))
            height, width, channel = resized_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(resized_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.image_label.setGeometry(150, 300, target_width, target_height) 
        else:
            print("Error: Unable to load the image. Check the file path and format.")


    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "All Files (*);;Image Files (*.png *.jpg *.jpeg)",
            options=options
        )

        if file_path:
            self.image_path = file_path
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi_color = img_color[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi_color, (48, 48))
                
                face_array = np.expand_dims(face_roi_resized, axis=0)
                face_array = face_array / 255.0  

                emotion_prediction = emotion_model.predict(face_array)[0]  
                emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                prediction_text = ""
                for idx, emotion in enumerate(emotions):
                    prediction_text += f"{emotion}: {emotion_prediction[idx] * 100:.2f}% "

                cv2.rectangle(img_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_color, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

            self.display_resized_image(img_color)


    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "All Files (*);;Video Files (*.mp4 *.avi *.mov)",
            options=options,
        )
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                print("Error: Unable to open the video file.")
                return
            self.video_playing = True
            self.timer.start(30)

    def select_camera(self):
        if self.video_playing:
            self.cap.release()
            self.video_playing = False
            self.timer.stop()

        self.cap = cv2.VideoCapture(0)
        self.video_playing = True
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) > 0:  
                faces_data = []
                for (x, y, w, h) in faces:
                    face_roi_color = frame[y:y+h, x:x+w]
                    face_roi_resized = cv2.resize(face_roi_color, (48, 48))
                    face_array = face_roi_resized / 255.0
                    faces_data.append(face_array)
                
                if faces_data:
                    faces_array = np.array(faces_data)
                    emotions_predictions = emotion_model.predict(faces_array)
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        emotion_label = np.argmax(emotions_predictions[i])
                        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                        predicted_emotion = emotions[emotion_label]

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_resized_image(frame_color)
        else:
            self.video_playing = False
            self.timer.stop()

    def keyPressEvent(self, event):
        if event.key() == 32 and self.video_playing:
            self.video_playing = False
            self.timer.stop()


if __name__ == "__main__":
    app = QApplication([])
    age_detection_ui = AgeDetectionUI()
    age_detection_ui.show()
    app.exec_()
