# coding: utf-8
import os
import cv2
import numpy as np
from util.show_image import *

# 脚本只用来处理田贵松提供的图片数据
org_im_root = '../data/tree/'
tmp_im_root = '../data/tree1/'
for img_id in os.listdir(org_im_root):
    im_path = org_im_root + img_id
    im = cv2.imread(im_path)
    im_h, im_w, _ = im.shape
    im_crop = im[int(im_h/3):, int(im_h/8):, :]

    # resize
    # im_h, im_w, _ = im_crop.shape
    # resize_ratio = 800.0 / im_h
    # im_h = int(im_h * resize_ratio)
    # im_w = int(im_w * resize_ratio)
    # im_crop = cv2.resize(im_crop, (im_w, im_h))

    im_output_path = tmp_im_root+img_id
    cv2.imwrite(im_output_path, im_crop.astype(np.uint8))
