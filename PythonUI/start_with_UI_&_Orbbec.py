import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import zipfile
from datetime import datetime
from pyorbbecsdk import *
from utils import frame_to_bgr_image

# Initialize variables
set_num = 1
pic_num = 1

# Create a RealSense pipeline
rs_pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
rs_config = rs.config()
rs_pipeline_wrapper = rs.pipeline_wrapper(rs_pipeline)
rs_pipeline_profile = rs_config.resolve(rs_pipeline_wrapper)
rs_device = rs_pipeline_profile.get_device()
rs_device_product_line = str(rs_device.get_info(rs.camera_info.product_line))

# Check if the RGB camera is found
found_rgb = False
for s in rs_device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Configure the depth and color streams for RealSense
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if rs_device_product_line == 'L500':
    rs_config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming for RealSense
rs_profile = rs_pipeline.start(rs_config)

# Get the depth sensor's depth scale for RealSense
rs_depth_sensor = rs_profile.get_device().first_depth_sensor()
rs_depth_scale = rs_depth_sensor.get_depth_scale()
print("RealSense Depth Scale is: ", rs_depth_scale)

# Define clipping distance for background removal for RealSense
rs_clipping_distance_in_meters = 1  # 1 meter
rs_clipping_distance = rs_clipping_distance_in_meters / rs_depth_scale

# Align depth frames to color frames for RealSense
rs_align_to = rs.stream.color
rs_align = rs.align(rs_align_to)

# Initialize Orbbec pipeline
orbbec_pipeline = Pipeline()
orbbec_config = Config()
orbbec_device = orbbec_pipeline.get_device()

# Configure the depth and color streams for Orbbec
orbbec_profile_list = orbbec_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
orbbec_color_profile = orbbec_profile_list.get_default_video_stream_profile()
orbbec_config.enable_stream(orbbec_color_profile)
orbbec_profile_list = orbbec_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
orbbec_depth_profile = orbbec_profile_list.get_default_video_stream_profile()
orbbec_config.enable_stream(orbbec_depth_profile)
orbbec_pipeline.start(orbbec_config)
orbbec_pipeline.enable_frame_sync()

# Align depth frames to color frames for Orbbec
orbbec_align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

# Initialize the regular camera
cap = cv2.VideoCapture(0)  # 0 is usually the index for the built-in camera

# Create directory for saving images and data
if not os.path.exists('saved_pics'):
    os.makedirs('saved_pics')

# Processing blocks for RealSense
rs_pc = rs.pointcloud()
rs_colorizer = rs.colorizer()
rs_decimate = rs.decimation_filter()

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

        rs_frames = rs_pipeline.wait_for_frames()
        rs_aligned_frames = rs_align.process(rs_frames)
        rs_aligned_depth_frame = rs_aligned_frames.get_depth_frame()
        rs_color_frame = rs_aligned_frames.get_color_frame()

        if not rs_aligned_depth_frame or not rs_color_frame:
            return

        orbbec_frames = orbbec_pipeline.wait_for_frames(100)
        orbbec_frames = orbbec_align_filter.process(orbbec_frames)
        orbbec_color_frame = orbbec_frames.get_color_frame()
        orbbec_depth_frame = orbbec_frames.get_depth_frame()

        if not orbbec_color_frame or not orbbec_depth_frame:
            return

        ret, webcam_frame = cap.read()
        if not ret:
            self.append_output("Failed to fetch frame from the webcam")
            return

        rs_depth_image = np.asanyarray(rs_aligned_depth_frame.get_data())
        rs_color_image = np.asanyarray(rs_color_frame.get_data())
        orbbec_color_image = frame_to_bgr_image(orbbec_color_frame)
        orbbec_depth_data = np.frombuffer(orbbec_depth_frame.get_data(), dtype=np.uint16).reshape(
            (orbbec_depth_frame.get_height(), orbbec_depth_frame.get_width()))
        orbbec_depth_image = cv2.normalize(orbbec_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        orbbec_depth_image = cv2.applyColorMap(orbbec_depth_image, cv2.COLORMAP_JET)

        rs_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(rs_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((rs_color_image, rs_depth_colormap, webcam_frame, orbbec_color_image, orbbec_depth_image))

        qimg = QtGui.QImage(images.data, images.shape[1], images.shape[0], QtGui.QImage.Format_BGR888)
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(qimg))

        if not state.paused:
            rs_depth_frame = rs_decimate.process(rs_aligned_depth_frame)
            rs_depth_intrinsics = rs.video_stream_profile(rs_depth_frame.profile).get_intrinsics()
            rs_depth_image = np.asanyarray(rs_depth_frame.get_data())
            rs_color_image = np.asanyarray(rs_color_frame.get_data())
            rs_depth_colormap = np.asanyarray(rs_colorizer.colorize(rs_depth_frame).get_data())

            if state.color:
                rs_mapped_frame, rs_color_source = rs_color_frame, rs_color_image
            else:
                rs_mapped_frame, rs_color_source = rs_depth_frame, rs_depth_colormap

            rs_points = rs_pc.calculate(rs_depth_frame)
            rs_pc.map_to(rs_mapped_frame)

            rs_v, rs_t = rs_points.get_vertices(), rs_points.get_texture_coordinates()
            rs_verts = np.asanyarray(rs_v).view(np.float32).reshape(-1, 3)
            rs_texcoords = np.asanyarray(rs_t).view(np.float32).reshape(-1, 2)

        out.fill(0)
        pointcloud(out, rs_verts, rs_texcoords, rs_color_source)
        qimg = QtGui.QImage(out.data, out.shape[1], out.shape[0], QtGui.QImage.Format_BGR888)
        self.pointcloud_canvas.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def save_data(self):
        global set_num, pic_num, out

        rs_frames = rs_pipeline.wait_for_frames()
        rs_aligned_frames = rs_align.process(rs_frames)
        rs_aligned_depth_frame = rs_aligned_frames.get_depth_frame()
        rs_color_frame = rs_aligned_frames.get_color_frame()

        if not rs_aligned_depth_frame or not rs_color_frame:
            return

        orbbec_frames = orbbec_pipeline.wait_for_frames(100)
        orbbec_frames = orbbec_align_filter.process(orbbec_frames)
        orbbec_color_frame = orbbec_frames.get_color_frame()
        orbbec_depth_frame = orbbec_frames.get_depth_frame()

        if not orbbec_color_frame or not orbbec_depth_frame:
            return

        ret, webcam_frame = cap.read()
        if not ret:
            self.append_output("Failed to fetch frame from the webcam")
            return

        rs_depth_image = np.asanyarray(rs_aligned_depth_frame.get_data())
        rs_color_image = np.asanyarray(rs_color_frame.get_data())
        orbbec_color_image = frame_to_bgr_image(orbbec_color_frame)
        orbbec_depth_data = np.frombuffer(orbbec_depth_frame.get_data(), dtype=np.uint16).reshape(
            (orbbec_depth_frame.get_height(), orbbec_depth_frame.get_width()))
        orbbec_depth_image = cv2.normalize(orbbec_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        orbbec_depth_image = cv2.applyColorMap(orbbec_depth_image, cv2.COLORMAP_JET)

        rs_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(rs_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        rs_depth_img_path = f'saved_pics/{set_num}_{pic_num}_RealSense_DepthHeatMap.png'
        rs_depth_array_path = f'saved_pics/{set_num}_{pic_num}_RealSense_DepthCamera.npy'
        rs_color_image_path = f'saved_pics/{set_num}_{pic_num}_RealSense_SideViewCamera.png'
        webcam_image_path = f'saved_pics/{set_num}_{pic_num}_Webcam_TopViewCamera.png'
        rs_pointcloud_image_path = f'saved_pics/{set_num}_{pic_num}_RealSense_PointCloud.png'

        orbbec_depth_img_path = f'saved_pics/{set_num}_{pic_num}_Orbbec_DepthHeatMap.png'
        orbbec_color_image_path = f'saved_pics/{set_num}_{pic_num}_Orbbec_SideViewCamera.png'
        orbbec_depth_array_path = f'saved_pics/{set_num}_{pic_num}_Orbbec_DepthCamera.npy'
        orbbec_pointcloud_image_path = f'saved_pics/{set_num}_{pic_num}_Orbbec_PointCloud.png'

        cv2.imwrite(rs_depth_img_path, rs_depth_colormap)
        cv2.imwrite(rs_color_image_path, rs_color_image)
        np.save(rs_depth_array_path, rs_depth_image)
        cv2.imwrite(webcam_image_path, webcam_frame)
        cv2.imwrite(rs_pointcloud_image_path, out)
        rs_points = rs_pc.calculate(rs_aligned_depth_frame)
        rs_pc.map_to(rs_color_frame)
        rs_points.export_to_ply(f'saved_pics/{set_num}_{pic_num}_RealSense_PointCloud.ply', rs_color_frame)
        print(f"RealSense point cloud data saved as {set_num}_{pic_num}_RealSense_PointCloud.ply")

        cv2.imwrite(orbbec_depth_img_path, orbbec_depth_image)
        cv2.imwrite(orbbec_color_image_path, orbbec_color_image)
        np.save(orbbec_depth_array_path, orbbec_depth_data)
        points = orbbec_frames.get_point_cloud(orbbec_pipeline.get_camera_param())
        if len(points) > 0:
            points_array = np.array([tuple(point) for point in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            orbbec_points_filename = f'saved_pics/{set_num}_{pic_num}_Orbbec_PointCloud.ply'
            el = PlyElement.describe(points_array, 'vertex')
            PlyData([el], text=True).write(orbbec_points_filename)
            print(f"Orbbec point cloud data saved as {orbbec_points_filename}")

        self.append_output(f"Data saved as {set_num}_{pic_num}")
        pic_num += 1

    def quit_app(self):
        self.timer.stop()
        rs_pipeline.stop()
        orbbec_pipeline.stop()
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
        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
