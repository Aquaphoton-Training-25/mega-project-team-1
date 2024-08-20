from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
import cv2
import os
from MainWindow_UI import Ui_MainWindow
from VideoStitching_backend import VideoStitching
from StereoVision_backend import StereoVision
from credits_backend import credits
import carControl
import serial
import time

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # connecting signals
        self.ui.start_pushButton.pressed.connect(self.start_camera_feed)
        self.ui.stop_pushButton.clicked.connect(self.stop_camera_feed)
        self.ui.screenshot_pushButton.clicked.connect(self.screen_shot)
        
        self.ui.videoStitching_pushButton.clicked.connect(self.open_videoStitching)
        self.ui.StereoVision_pushButton.clicked.connect(self.open_StereoVision)
        self.ui.credits_pushButton.clicked.connect(self.open_Credits)
        
        self.ui.autonomus_mode_pushButton.clicked.connect(self.set_mode_auto)
        self.ui.manual_mode_pushButton.clicked.connect(self.set_mode_manual)
        
        self.ui.high_speed_pushButton.clicked.connect(self.set_speed_high)
        self.ui.medium_speed_pushButton.clicked.connect(self.set_speed_medium)
        self.ui.low_speed_pushButton.clicked.connect(self.set_speed_low)

        self.ui.forward_pushButton.pressed.connect(self.start_moving_forward)
        self.ui.forward_pushButton.released.connect(self.stop_moving_forward)
        self.ui.backward_pushButton.pressed.connect(self.start_moving_backward)
        self.ui.backward_pushButton.released.connect(self.stop_moving_backward)
        self.ui.right_pushButton.pressed.connect(self.start_moving_right)
        self.ui.right_pushButton.released.connect(self.stop_moving_right)
        self.ui.left_pushButton.pressed.connect(self.start_moving_left)
        self.ui.left_pushButton.released.connect(self.stop_moving_left)
        
        self.ui.keyboard_pushButton.clicked.connect(self.setup_keyboard)
        self.ui.joystick_pushButton.clicked.connect(self.control_by_GUI)

        # Keep track of dialog windows to avoid opening multiple instances
        self.video_stitching_window = None
        self.stereo_vision_window = None
        self.credits = None

        # Variable for checking the keyboard
        self.iskeyboard = False # set it false to make the GUI the default control

        # Initialize the serial connection (Make sure the port is correct)
        arduino_port = "COM3"  # Update with your port name (e.g., COM3 for Windows, /dev/ttyUSB0 for Linux)
        baud_rate = 9600

        self.ser = serial.Serial(arduino_port, baud_rate, timeout=1)

        time.sleep(2)  # Wait for the Arduino to reset

        # Timer to simulate continuous movement
        self.forward_timer = QTimer()
        self.forward_timer.timeout.connect(self.forward)
        self.backward_timer = QTimer()
        self.backward_timer.timeout.connect(self.backward)
        self.right_timer = QTimer()
        self.right_timer.timeout.connect(self.right)
        self.left_timer = QTimer()
        self.left_timer.timeout.connect(self.left)

        # Install event filter
        self.installEventFilter(self)

    def control_by_GUI(self):
        self.iskeyboard = False
        self.ui.forward_pushButton.setEnabled(True)
        self.ui.backward_pushButton.setEnabled(True)
        self.ui.right_pushButton.setEnabled(True)
        self.ui.left_pushButton.setEnabled(True)
    
    def receive_data(self):
        """Receives and parses sensor data from the Arduino."""
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8').strip("\r\n") #if not convenient convert to utf-8 using str method
            if data:
                # Data format: voltage:current:distance
                try:
                    voltage, current, distance, state= data.split(':')
                    print(f"Voltage: {voltage} V, Current: {current} A, Distance: {distance} cm, State={state}")
                except ValueError:
                    print("Failed to parse sensor data")

    def setup_keyboard(self):
        self.iskeyboard = True
        print("Keyboard control enabled")
        self.ui.forward_pushButton.setDisabled(True)
        self.ui.backward_pushButton.setDisabled(True)
        self.ui.right_pushButton.setDisabled(True)
        self.ui.left_pushButton.setDisabled(True)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and self.iskeyboard:
            if event.key() == Qt.Key.Key_W:
                print("w is pressed")
                self.start_moving_forward()
            elif event.key() == Qt.Key.Key_S:
                print("s is pressed")
                self.start_moving_backward()
            elif event.key() == Qt.Key.Key_D:
                print("d is pressed")
                self.start_moving_right()
            elif event.key() == Qt.Key.Key_A:
                print("a is pressed")
                self.start_moving_left()
        elif event.type() == QEvent.Type.KeyRelease and self.iskeyboard:
            if event.key() == Qt.Key.Key_W:
                print("w is released")
                self.stop_moving_forward()
            elif event.key() == Qt.Key.Key_S:
                print("s is released")
                self.stop_moving_backward()
            elif event.key() == Qt.Key.Key_D:
                print("d is pressed")
                self.stop_moving_right()
            elif event.key() == Qt.Key.Key_A:
                print("a is pressed")
                self.stop_moving_left()
        return super(Window, self).eventFilter(source, event)

    def start_moving_forward(self):
        self.forward_timer.start(100)  # Call forward() every 100ms

    def stop_moving_forward(self):
        self.forward_timer.stop()

    def forward(self):
        print("Moving forward")
        carControl.forward()
        self.ui.direction_reading.setText("F")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def start_moving_backward(self):
        self.backward_timer.start(100)
        

    def stop_moving_backward(self):
        self.backward_timer.stop()
        

    def backward(self):
        print("Moving backward")
        carControl.backward()
        self.ui.direction_reading.setText("B")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def start_moving_right(self):
        self.right_timer.start(100)
        

    def stop_moving_right(self):
        self.right_timer.stop()

    def right(self):
        print("Moving right")
        carControl.move_right()
        self.ui.direction_reading.setText("R")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def start_moving_left(self):
        self.left_timer.start(100)
        

    def stop_moving_left(self):
        self.left_timer.stop()

    def left(self):
        print("Moving left")
        carControl.move_left()
        self.ui.direction_reading.setText("L")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

        

    def set_mode_auto(self):
        carControl.autonomous()
        self.ui.mode_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.mode_reading.setText("A")

    def set_mode_manual(self):
        self.ui.mode_reading.setStyleSheet("background-color:rgb(255,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.mode_reading.setText("M")

    def set_speed_high(self):
        carControl.high_speed()
        self.ui.speed_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("H")
    
    def set_speed_medium(self):
        carControl.medium_speed()
        self.ui.speed_reading.setStyleSheet("background-color:rgb(255,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("M")

    def set_speed_low(self):
        carControl.low_speed()
        self.ui.speed_reading.setStyleSheet("background-color:rgb(255,0,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("L")    

    def start_camera_feed(self):
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.update_feed)

    def update_feed(self, Image):
        self.ui.camera_label.setPixmap(QPixmap.fromImage(Image))

    def stop_camera_feed(self):
        if hasattr(self, 'Worker1'):
            self.Worker1.stop()  # Gracefully stop the thread
            self.Worker1.quit()  # Ensure it quits after stopping
            self.Worker1.wait()  # Wait for the thread to finish
            self.Worker1.ImageUpdate.disconnect()  # Disconnect signal after stopping
            self.clear_feed()  # Clear the camera label


    def clear_feed(self):
        self.ui.camera_label.clear()

    def screen_shot(self):
        if hasattr(self, 'Worker1'):
            self.Worker1.take_screenshot()  # Trigger screenshot in the worker thread

    def open_videoStitching(self):
        # Check if the dialog is already open
        if self.video_stitching_window is None:
            # Instantiate the video stitching dialog
            self.video_stitching_window = VideoStitching()

        if not self.video_stitching_window.isVisible():
            self.video_stitching_window.show()
        else:
            self.video_stitching_window.raise_()  # Bring the dialog to the front

    def open_StereoVision(self):
        # Check if the dialog is already open
        if self.stereo_vision_window is None:
            # Instantiate the Stereo Vision dialog
            self.stereo_vision_window = StereoVision()

        if not self.stereo_vision_window.isVisible():
            self.stereo_vision_window.show()
        else:
            self.stereo_vision_window.raise_()  # Bring the dialog to the front

    def open_Credits(self):
        # Check if the dialog is already open
        if self.credits is None:
            # Instantiate the Credits dialog
            self.credits = credits()

        if not self.credits.isVisible():
            self.credits.show()
        else:
            self.credits.raise_()  # Bring the dialog to the front

    def closeEvent(self, event):
        # Clean up dialog windows when the main window is closed
        if self.video_stitching_window:
            self.video_stitching_window.close()
        if self.stereo_vision_window:
            self.stereo_vision_window.close()
        if self.credits:
            self.credits.close()
        event.accept()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.ThreadActive = True
        self.screenshot_requested = False  # Flag for screenshot
        self.capture = None  # Initialize the video capture object

    def run(self):
        self.ThreadActive = True
        self.capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = self.capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

                # Emit the image for the live feed
                self.ImageUpdate.emit(Pic)

                # Save screenshot if requested
                if self.screenshot_requested:
                    self.save_screenshot(frame)
                    self.screenshot_requested = False

        # Release resources after the loop ends
        self.capture.release()

    def take_screenshot(self):
        """Method to request a screenshot."""
        self.screenshot_requested = True

    def save_screenshot(self, frame):
        """Method to save the screenshot."""
        screenshot_dir = 'screenshots'
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        filename = os.path.join(screenshot_dir, 'screenshot.png')
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def stop(self):
        """Stop the thread and ensure a clean shutdown."""
        self.ThreadActive = False
        self.wait()  # Wait for the thread to properly exit before quitting



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
