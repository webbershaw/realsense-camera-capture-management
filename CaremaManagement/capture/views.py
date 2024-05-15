import os

from django.shortcuts import render
from django.http import HttpResponse
import cv2
from django.http import StreamingHttpResponse
from django.http import JsonResponse
import zipfile
from . import realsense
import pyrealsense2 as rs
import numpy as np
from concurrent.futures import ThreadPoolExecutor

deviceList = {}

suc, fr = realsense.read_depth_frame()
print(fr)


def list_cameras(max_cameras=9):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    if realsense.isConnected():
        available_cameras.append(9)
    return available_cameras


def index(request):
    return HttpResponse("Hello, World")


def gen(camera):
    i = 10
    while True:
        if camera == 9:
            break
        else:
            success, frame = camera.read()  # 从摄像头读取每一帧

        if not success:
            i = i - 1
            if i < 0:
                break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 拼接帧数据


def camera_stream(request):
    camera = int(request.GET.get('camera'))
    if camera == 9:
        return StreamingHttpResponse(gen(camera),
                                     content_type='multipart/x-mixed-replace; boundary=frame')

    camera = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

    # 尝试开启自动对焦
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1 代表开启自动对焦
    else:
        return HttpResponse("Cannot open camera")

    return StreamingHttpResponse(gen(camera),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def init_system(request):
    camera = request.GET.get('camera')
    tag = request.GET.get('tag')
    deviceList[int(camera)] = tag
    return HttpResponse('Success!')


def get_camera_list(request):
    return HttpResponse(list_cameras())


def get_device_list(request):
    print(deviceList)
    return JsonResponse(deviceList)


def capture_images(request):
    name = request.GET.get('name', 'default')  # 从 GET 请求获取 'name' 参数

    with ThreadPoolExecutor() as executor:
        futures = []

        for device_key, device_name in deviceList.items():
            if device_key == 9:
                filename = f"{name}_{device_name}"
                futures.append(executor.submit(realsense.save_depth_data_and_image, filename=filename))
                continue

            futures.append(executor.submit(process_frame, device_key, device_name, name))

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    return HttpResponse("Images captured")


def process_frame(device_key, device_name, name):
    cap = cv2.VideoCapture(device_key, cv2.CAP_DSHOW)  # 用设备键值打开摄像头
    success, frame = cap.read()  # 读取当前帧
    cap.release()  # 释放摄像头

    if success:
        # 构建文件名和路径
        filename = f"{name}_{device_name}.jpg"

        # 保存图像
        cv2.imwrite('static/data/' + filename, frame)


def export(request):
    folder_path = 'static/data/'  # 设置要压缩的文件夹路径
    zip_filename = 'dataset.zip'  # 压缩文件的名称

    # 创建一个临时的压缩文件
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file))

    # 读取压缩文件并准备响应
    with open(zip_filename, 'rb') as f:
        response = HttpResponse(f, content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename={zip_filename}'
        return response


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):  # 如果需要删除子文件夹取消注释
            # shutil.rmtree(file_path)
            pass


def clear(request):
    clear_folder('static/data/')
    return HttpResponse('Cleared!')
