# coding: utf-8
import os
# 用于保存中间结果
TEMP_PATH = os.path.join(os.path.dirname(__file__), 'vis')


POINT_DISTANCE = 120    # mm
ORG_IM_HEIGHT = 4000    # pixel
ORG_IM_WIDTH = 3000     # pixel

# TODO: ----- 注意 -----
# 如果改变IMG_MAX_HEIGHT，FOCAL_LENGTH需要重新估计
# 估计FOCAL_LENGTH: 运行 prepare/camera.py
IMG_MAX_HEIGHT = 800        # pixel
FOCAL_LENGTH = 591          # pixel

# IMG_MAX_HEIGHT = 2000
# FOCAL_LENGTH = -1


# IMG_MAX_HEIGHT = 4000
# FOCAL_LENGTH = 3173
# TODO: ----- 注意 -----


# 分割网络能接受的最大图像宽度
# 更换显卡时需重新测试并设置
# ( 6G GTX1060): 800
# (12G GTX1080): ?
NET_MAX_WIDTH = 800



