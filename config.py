# coding: utf-8
TAG_WIDTH = 27          # mm
TAG_HEIGHT = 91         # mm


# 注意：
# 如果改变IMG_MAX_HEIGHT，FOCAL_LENGTH需要重新估计
# 估计FOCAL_LENGTH: 运行 prepare/camera.py
IMG_MAX_HEIGHT = 800    # pixel
FOCAL_LENGTH = 697      # pixel

# IMG_MAX_HEIGHT = 2000
# FOCAL_LENGTH = -1


# IMG_MAX_HEIGHT = 4000
# FOCAL_LENGTH = 3173


# 分割网络能接受的最大图像宽度
# 更换显卡时需重新测试并设置
# ( 6G GTX1060): 800
# (12G GTX1080): ?
NET_MAX_WIDTH = 800
