from django.urls import path

from . import views  # 导入当级目录的views.py文件

urlpatterns = [
    # path("", views.index, name="index"),
    path("video", views.camera_stream, name="camera_stream"),
    path("list/", views.get_camera_list, name="get_camera_list"),
    path("device", views.get_device_list, name="get_device_list"),
    path("bind", views.init_system, name="init"),
    path("cap", views.capture_images, name="cap"),
    path("export", views.export, name="export"),
    path("clear", views.clear, name="clear"),
]
