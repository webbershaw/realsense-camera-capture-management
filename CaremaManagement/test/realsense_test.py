import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

duck_num = 1
pic_num = 1

# 创建RealSense管道
pipeline = rs.pipeline()

# 创建配置并配置管道以进行流式传输
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# 检查是否找到了RGB摄像头
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 配置深度和颜色流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


profile = pipeline.start(config)

# 获取深度传感器的深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# 用于背景移除的裁剪距离
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# 对齐深度帧和颜色帧
align_to = rs.stream.color
align = rs.align(align_to)

# 初始化普通摄像头
cap = cv2.VideoCapture(1)  # 0通常是内置摄像头的索引

# 创建保存图像和数据的目录
if not os.path.exists('saved_pics'):
    os.makedirs('saved_pics')

try:
    while True:
        # 从RealSense设备获取帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 从普通摄像头获取帧
        ret, webcam_frame = cap.read()
        if not ret:
            print("Failed to fetch frame from the webcam")
            break

        # 处理深度帧
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # 为深度图像应用颜色映射
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 将RealSense的图像和普通摄像头的图像水平拼接
        images = np.hstack((bg_removed, depth_colormap, webcam_frame))

        # 显示图像
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
              # 获取用户输入的文件名
            depth_image_path = f'saved_pics/'+str(duck_num)+'_'+str(pic_num)+'_bgremoved.png'
            depth_img_path = f'saved_pics/'+str(duck_num)+'_'+str(pic_num)+'_depth.png'
            depth_array_path = f'saved_pics/'+str(duck_num)+'_'+str(pic_num)+'_depth_array'
            color_image_path = f'saved_pics/'+str(duck_num)+'_'+str(pic_num)+'_color.png'
            webcam_image_path = f'saved_pics/'+str(duck_num)+'_'+str(pic_num)+'_webcam.png'

            # 保存图像和数组
            cv2.imwrite(depth_image_path, bg_removed)
            cv2.imwrite(depth_img_path, depth_image)
            cv2.imwrite(color_image_path, color_image)
            np.save(depth_array_path, depth_image)
            cv2.imwrite(webcam_image_path, webcam_frame)

            print(f"Data saved as "+str(duck_num)+'_'+str(pic_num))
            pic_num = pic_num +1


        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        elif key & 0xFF == ord('b'):
            pic_num = pic_num - 1;
            print('New pic num is '+str(pic_num))

        elif key & 0xFF == ord('n'):
            duck_num = duck_num - 1;
            print('New duck num is '+str(duck_num))

        elif key & 0xFF == ord('p'):
            duck_num = duck_num+1
            pic_num = 1
            print('New duck num' + str(duck_num))

finally:
    pipeline.stop()
    cap.release()
