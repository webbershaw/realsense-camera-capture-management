import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math
import time

# Initialize variables
duck_num = 1
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


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:
        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


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


out = np.empty((480, 640, 3), dtype=np.uint8)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        ret, webcam_frame = cap.read()
        if not ret:
            print("Failed to fetch frame from the webcam")
            break

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap, webcam_frame))

        cv2.namedWindow('Camera Monitor', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera Monitor', images)

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

        now = time.time()
        out.fill(0)

        pointcloud(out, verts, texcoords, color_source)

        cv2.imshow(state.WIN_NAME, out)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            depth_img_path = f'saved_pics/{duck_num}_{pic_num}_DepthHeatMap.png'
            depth_array_path = f'saved_pics/{duck_num}_{pic_num}_DepthCamera.npy'
            color_image_path = f'saved_pics/{duck_num}_{pic_num}_SideViewCamera.png'
            webcam_image_path = f'saved_pics/{duck_num}_{pic_num}_TopViewCamera.png'
            pointcloud_image_path = f'saved_pics/{duck_num}_{pic_num}_PointCloud.png'


            cv2.imwrite(depth_img_path, depth_colormap)
            cv2.imwrite(color_image_path, color_image)
            np.save(depth_array_path, depth_image)
            cv2.imwrite(webcam_image_path, webcam_frame)
            cv2.imwrite(pointcloud_image_path, out)
            points.export_to_ply(f'saved_pics/{duck_num}_{pic_num}_PointCloud.ply', mapped_frame)
            print(f"Point cloud data saved as {duck_num}_{pic_num}_PointCloud.ply")

            print(f"Data saved as {duck_num}_{pic_num}")
            pic_num += 1

        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        elif key & 0xFF == ord('b'):
            pic_num = max(1, pic_num - 1)
            print(f'New pic num is {pic_num}')

        elif key & 0xFF == ord('n'):
            duck_num = max(1, duck_num - 1)
            print(f'New duck num is {duck_num}')

        elif key & 0xFF == ord('p'):
            duck_num += 1
            pic_num = 1
            print(f'New duck num is {duck_num}')





finally:
    pipeline.stop()
    cap.release()


