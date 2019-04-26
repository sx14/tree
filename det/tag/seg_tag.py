# coding: utf-8
import os
import cv2
import numpy as np
from util.show_image import *


def segment_tag(im):
    im_h, im_w, _ = im.shape
    im = im.astype(np.int32)
    # im = cv2.resize(im, (int(im_w * 0.1), int(im_h * 0.1)))
    im_bin = im[:, :, 0] - im[:, :, 1] - im[:, :, 2]
    im_bin[im_bin < 20] = 0
    im_bin[im_bin > 0] = 255
    im_bin = im_bin.astype(np.uint8)
    # 对图像进行“开运算”
    # 先腐蚀
    kernel = np.ones((3, 3), np.uint8)
    im_bin1 = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel, iterations=4)
    # 再膨胀
    im_bin2 = cv2.dilate(im_bin1, kernel, iterations=4)
    return im_bin2
