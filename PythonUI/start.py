import pyrealsense2 as rs
import numpy as np
import cv2
import os

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

try:
    while True:
        # Get frames from the RealSense device
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Get frames from the regular camera
        ret, webcam_frame = cap.read()
        if not ret:
            print("Failed to fetch frame from the webcam")
            break

        # Process depth frame
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to the depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack images horizontally
        images = np.hstack((color_image, depth_colormap, webcam_frame))

        # Display the images
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            # Define paths for saving images and data
            depth_img_path = f'saved_pics/{duck_num}_{pic_num}_DepthHeatMap.png'
            depth_array_path = f'saved_pics/{duck_num}_{pic_num}_DepthCamera.npy'
            color_image_path = f'saved_pics/{duck_num}_{pic_num}_SideViewCamera.png'
            webcam_image_path = f'saved_pics/{duck_num}_{pic_num}_TopViewCamera.png'

            # Save images and depth data
            cv2.imwrite(depth_img_path, depth_colormap)
            cv2.imwrite(color_image_path, color_image)
            np.save(depth_array_path, depth_image)
            cv2.imwrite(webcam_image_path, webcam_frame)

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
