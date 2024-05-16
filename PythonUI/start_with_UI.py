import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import zipfile
from datetime import datetime

# Initialize variables
set_num = 1
pic_num = 1

# Create a RealSense pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Check if the RGB camera is found
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Configure the depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Define clipping distance for background removal
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize the regular camera
cap = cv2.VideoCapture(0)  # 0 is usually the index for the built-in camera

# Create directory for saving images and data
if not os.path.exists('saved_pics'):
    os.makedirs('saved_pics')

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()
decimate = rs.decimation_filter()

# Class to manage application state
class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'Point Cloud Monitor'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

out = np.empty((480, 640, 3), dtype=np.uint8)

def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (w / 2.0, h / 2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj

def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]

    j, i = proj.astype(np.uint32).T

    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T

    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    out[i[m], j[m]] = color[u[m], v[m]]

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RealSense and Camera Capture Management")
        self.setGeometry(100, 100, 1400, 800)

        self.layout = QtWidgets.QVBoxLayout()

        self.canvas = QtWidgets.QLabel()
        self.layout.addWidget(self.canvas)

        self.pointcloud_canvas = QtWidgets.QLabel()
        self.layout.addWidget(self.pointcloud_canvas)

        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        self.buttons_layout = QtWidgets.QHBoxLayout()

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.save_data)
        self.buttons_layout.addWidget(self.save_button)

        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.clicked.connect(self.export)
        self.buttons_layout.addWidget(self.export_button)

        self.quit_button = QtWidgets.QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit_app)
        self.buttons_layout.addWidget(self.quit_button)

        self.prev_pic_button = QtWidgets.QPushButton("Previous Picture")
        self.prev_pic_button.clicked.connect(self.decrement_pic_num)
        self.buttons_layout.addWidget(self.prev_pic_button)

        self.prev_set_button = QtWidgets.QPushButton("Previous Set")
        self.prev_set_button.clicked.connect(self.decrement_set_num)
        self.buttons_layout.addWidget(self.prev_set_button)

        self.next_set_button = QtWidgets.QPushButton("Next Set")
        self.next_set_button.clicked.connect(self.increment_set_num)
        self.buttons_layout.addWidget(self.next_set_button)

        self.layout.addLayout(self.buttons_layout)

        self.textboxes_layout = QtWidgets.QHBoxLayout()

        self.set_num_label = QtWidgets.QLabel("Set Num:")
        self.textboxes_layout.addWidget(self.set_num_label)
        self.set_num_textbox = QtWidgets.QLineEdit(str(set_num))
        self.textboxes_layout.addWidget(self.set_num_textbox)
        self.set_num_button = QtWidgets.QPushButton("Set Set Num")
        self.set_num_button.clicked.connect(self.set_set_num)
        self.textboxes_layout.addWidget(self.set_num_button)

        self.pic_num_label = QtWidgets.QLabel("Pic Num:")
        self.textboxes_layout.addWidget(self.pic_num_label)
        self.pic_num_textbox = QtWidgets.QLineEdit(str(pic_num))
        self.textboxes_layout.addWidget(self.pic_num_textbox)
        self.pic_num_button = QtWidgets.QPushButton("Set Pic Num")
        self.pic_num_button.clicked.connect(self.set_pic_num)
        self.textboxes_layout.addWidget(self.pic_num_button)

        self.layout.addLayout(self.textboxes_layout)

        self.setLayout(self.layout)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        global set_num, pic_num, out

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return

        ret, webcam_frame = cap.read()
        if not ret:
            self.append_output("Failed to fetch frame from the webcam")
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap, webcam_frame))

        qimg = QtGui.QImage(images.data, images.shape[1], images.shape[0], QtGui.QImage.Format_BGR888)
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(qimg))

        if not state.paused:
            depth_frame = decimate.process(aligned_depth_frame)
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

        out.fill(0)
        pointcloud(out, verts, texcoords, color_source)
        qimg = QtGui.QImage(out.data, out.shape[1], out.shape[0], QtGui.QImage.Format_BGR888)
        self.pointcloud_canvas.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def save_data(self):
        global set_num, pic_num, out

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return

        ret, webcam_frame = cap.read()
        if not ret:
            self.append_output("Failed to fetch frame from the webcam")
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_img_path = f'saved_pics/{set_num}_{pic_num}_DepthHeatMap.png'
        depth_array_path = f'saved_pics/{set_num}_{pic_num}_DepthCamera.npy'
        color_image_path = f'saved_pics/{set_num}_{pic_num}_SideViewCamera.png'
        webcam_image_path = f'saved_pics/{set_num}_{pic_num}_TopViewCamera.png'
        pointcloud_image_path = f'saved_pics/{set_num}_{pic_num}_PointCloud.png'

        cv2.imwrite(depth_img_path, depth_colormap)
        cv2.imwrite(color_image_path, color_image)
        np.save(depth_array_path, depth_image)
        cv2.imwrite(webcam_image_path, webcam_frame)
        cv2.imwrite(pointcloud_image_path, out)

        # Calculate and export point cloud
        points = pc.calculate(aligned_depth_frame)
        pc.map_to(color_frame)
        points.export_to_ply(f'saved_pics/{set_num}_{pic_num}_PointCloud.ply', color_frame)
        self.append_output(f"Data saved as {set_num}_{pic_num}")

        pic_num += 1

    def quit_app(self):
        self.timer.stop()
        pipeline.stop()
        cap.release()
        QtWidgets.QApplication.quit()

    def decrement_pic_num(self):
        global pic_num
        pic_num = max(1, pic_num - 1)
        self.pic_num_textbox.setText(str(pic_num))

    def decrement_set_num(self):
        global set_num
        set_num = max(1, set_num - 1)
        self.set_num_textbox.setText(str(set_num))

    def increment_set_num(self):
        global set_num, pic_num
        set_num += 1
        self.set_num_textbox.setText(str(set_num))
        pic_num = 1
        self.pic_num_textbox.setText(str(pic_num))

    def set_set_num(self):
        global set_num
        try:
            set_num = int(self.set_num_textbox.text())
        except ValueError:
            self.set_num_textbox.setText(str(set_num))

    def set_pic_num(self):
        global pic_num
        try:
            pic_num = int(self.pic_num_textbox.text())
        except ValueError:
            self.pic_num_textbox.setText(str(pic_num))

    def export(self):
        if not os.path.exists('exports'):
            os.makedirs('exports')
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = os.path.join('exports', current_time + '_export.zip')
        with zipfile.ZipFile(output_filename, 'w') as zipf:
            for root, dirs, files in os.walk('saved_pics'):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, 'saved_pics'))

        self.append_output(f'Package Done! Export Path：{output_filename}')

    def append_output(self, text):
        self.output_text.append(text)
        self.output_text.ensureCursorVisible()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
