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

#import face_recognition
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
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

        self.detect_age_button = QPushButton("Detect Emotion", self)
        self.detect_age_button.setGeometry(400, 250, 100, 30)

        self.select_image_button.clicked.connect(self.select_image)
        self.select_video_button.clicked.connect(self.select_video)
        self.select_camera_button.clicked.connect(self.select_camera)
        self.detect_age_button.clicked.connect(self.detect_age)

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
            self.image_label.setGeometry(150, 300, target_width, target_height)  # Adjust position as needed
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
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert the image to display in PyQt
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_resized_image(img)


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
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert the frame to RGB for display in PyQt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_resized_image(frame_rgb)
        else:
            self.video_playing = False
            self.timer.stop()

    def detect_age(self):
        if self.faces is not None:
            # age detection here

            qp = QPainter(self.image_label.pixmap())
            pen = QPen(Qt.white, 4)
            qp.setPen(pen)
            font = QFont()
            font.setFamily("Times")
            font.setBold(True)
            font.setPointSize(20)
            qp.setFont(font)
            for x, y, w, h in self.faces:
                qp.drawText(x, y + h + 30, "Emotion: ")

            qp.end()

            age_result_label = QLabel(self)
            age_result_label.setGeometry(200, 460, 100, 30)
            age_result_label.setText("Emotioon: ")  

    def keyPressEvent(self, event):
        if event.key() == 32 and self.video_playing:
            self.video_playing = False
            self.timer.stop()


if __name__ == "__main__":
    app = QApplication([])
    age_detection_ui = AgeDetectionUI()
    age_detection_ui.show()
    app.exec_()
