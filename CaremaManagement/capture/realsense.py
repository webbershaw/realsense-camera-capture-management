import pyrealsense2 as rs

# 创建一个上下文对象
context = rs.context()

# 获取连接的设备列表
device_list = context.query_devices()


def isConnected():
    # 检查是否有设备在线
    if len(device_list) > 0:
        print("RealSense Depth Camera Detected!")
        return True

    else:
        print("No RealSense Depth Camera Alive!")
        return False


import pyrealsense2 as rs

import pyrealsense2 as rs
import numpy as np


def read_depth_frame():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        return False, None

    try:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            print("No depth frame available")
            return False, None

        # 将深度帧转换为 NumPy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        return True, depth_image
    finally:
        pipeline.stop()


import cv2


def main():
    success, depth_image = read_depth_frame()
    if success:
        # 将深度图像转换为伪彩色以便更好地可视化
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示深度图像
        cv2.imshow('Depth Frame', depth_colormap)

        # 按任意键退出
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to read depth frame")


import pyrealsense2 as rs
import numpy as np
import cv2


def save_depth_data_and_image(filename, depth_min=0.4, depth_max=1.2, folder='static/data/'):
    # Ensure the folder exists
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    # Configure depth stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("Could not obtain depth frame")

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth from millimeters to meters
        depth_image_meters = depth_image * 0.001

        # Apply distance threshold
        filtered_depth = np.where((depth_image_meters >= depth_min) & (depth_image_meters <= depth_max),
                                  depth_image_meters, 0)

        # Save depth data to numpy file
        npy_file = folder + filename + '.npy'
        np.save(npy_file, filtered_depth)

        # Normalize the depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255 / depth_frame.get_units()),
                                           cv2.COLORMAP_JET)

        # Save depth image to jpg file
        jpg_file = folder + filename + '.jpg'
        cv2.imwrite(jpg_file, depth_colormap)

    finally:
        # Stop streaming
        pipeline.stop()

# save_depth_data_and_image('aaa2')
# #
# #
# save_depth_data_and_image('aaa3')
# save_depth_data_and_image('aaa4')
