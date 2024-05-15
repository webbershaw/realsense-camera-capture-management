import threading
import cv2
class CameraManager:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            _, frame = self.camera.read()
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.camera.release()
