from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
import cv2
import os
from MainWindow_UI import Ui_MainWindow
from VideoStitching_backend import VideoStitching

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
        self.ui.autonomus_mode_pushButton.clicked.connect(self.set_mode_auto)
        self.ui.manual_mode_pushButton.clicked.connect(self.set_mode_manual)
        self.ui.high_speed_pushButton.clicked.connect(self.set_speed_high)
        self.ui.medium_speed_pushButton.clicked.connect(self.set_speed_medium)
        self.ui.low_speed_pushButton.clicked.connect(self.set_speed_low)

        # Keep track of video stitching window to avoid opening multiple instances
        self.video_stitching_window = None


    def set_mode_auto(self):
        self.ui.mode_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.mode_reading.setText("A")

    def set_mode_manual(self):
        self.ui.mode_reading.setStyleSheet("background-color:rgb(255,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.mode_reading.setText("M")

    def set_speed_high(self):
        self.ui.speed_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("H")
    
    def set_speed_medium(self):
        self.ui.speed_reading.setStyleSheet("background-color:rgb(255,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("M")

    def set_speed_low(self):
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

    def closeEvent(self, event):
        # Clean up video stitching window when the main window is closed
        if self.video_stitching_window:
            self.video_stitching_window.close()
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
