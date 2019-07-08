# coding: utf-8
import os

CALIBRATOR = 'laser'
# CALIBRATOR = 'tag'

# ===== tag =====
TAG_WIDTH = 27      # mm
TAG_HEIGHT = 91     # mm
# ===============


# ===== laser =====
POINT_DISTANCE = 120    # mm

# 改变IMG_MAX_HEIGHT，FOCAL_LENGTH需要重新估计
# 估计FOCAL_LENGTH: 运行 prepare/camera.py
IMG_MAX_HEIGHT = 1000        # pixel
FOCAL_LENGTH = 591           # pixel
# ==================


# ===== segmentation =====
# 分割网络能接受的最大图像宽度
# 更换显卡时需重新测试并设置
NET_MAX_WIDTH = 600
# ==================


# ===== resize =====
SAVE_WIDTH = 300
SAVE_HEIGHT = 300
# ==================

# 用于保存中间结果
PROJECT_ROOT = os.path.join(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


