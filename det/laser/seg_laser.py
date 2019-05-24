# coding: utf-8
import numpy as np
import cv2

from util.show_image import *
from config import *


def get_bright_pt_mask(im, laser_mask, show=False):
    # binary mat
    # 激光点区域
    # 排除激光线

    # 膨胀红色区域，使其能够覆盖高亮点
    kernel = np.ones((3, 3), np.uint8)
    laser_mask_d = cv2.dilate(laser_mask, kernel, iterations=5)

    bright_mask = get_bright(im, False)
    pt_mask = bright_mask & laser_mask_d

    # 膨胀高亮点，使其连成区域
    kernel = np.ones((3, 3), np.uint8)
    pt_mask = cv2.dilate(pt_mask.astype(np.uint8), kernel)

    if show:
        im_pt = pt_mask.astype(np.uint8)
        im_pt[im_pt > 0] = 255
        show_image(im_pt)

    return pt_mask


def get_bright(im, show=False):
    # binary mat
    # 高亮区域
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bright_mask = im_grey > 220

    # 膨胀高亮点，使其连成区域
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.dilate(bright_mask.astype(np.uint8), kernel)

    if show:
        im_bright = bright_mask.astype(np.uint8)
        im_bright[im_bright > 0] = 255
        show_images([im_bright])

    return bright_mask


def get_laser_mask(im, show=False):
    # binary mat
    # 红色区域
    im = im.astype(np.int32)
    laser_mask = im[:, :, 2] - np.max(im[:, :, [0, 1]], axis=2)
    laser_mask = laser_mask > 50
    laser_mask = laser_mask.astype(np.uint8)

    # 闭运算，去空洞
    kernel = np.ones((3, 3), np.uint8)
    laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=5)

    if show:
        im_red = laser_mask.astype(np.uint8)
        im_red[im_red > 0] = 255
        show_image(im_red)

    return laser_mask


def segment_laser_pts(im, calibrate=False):

    if calibrate:
        # 标定时使用极亮点
        laser_mask = get_bright(im, False)
        _, pt_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask)
        pt_mask = laser_mask.copy()
    else:
        # 测试时使用激光点
        laser_mask = get_laser_mask(im, False)
        _, pt_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask)
        pt_mask = laser_mask.copy()

    # 找两个激光点
    pt_ind_pairs = []
    im_h, im_w = pt_mask.shape
    im_center_x = im_w / 2.0
    im_center_y = im_h / 2.0
    for i in range(1, len(centroids)-1):
        for j in range(i+1, len(centroids)):
            cent1 = centroids[i]
            cent2 = centroids[j]

            area1 = stats[i][-1]
            area2 = stats[j][-1]

            w1, h1 = stats[i][2:4]
            w2, h2 = stats[j][2:4]
            wh1 = max(w1, h1) / min(w1, h1)
            wh2 = max(w2, h2) / min(w2, h2)

            # 两点水平方向距离
            x_diff = abs(cent1[0] - cent2[0])
            # 两点竖直方向距离
            y_diff = abs(cent1[1] - cent2[1])

            if not calibrate:
                # 1.两点都在右侧 TODO: 左侧
                # 2.两点在中线一上一下 TODO: 中间区域
                # 3.两点x坐标相差不大，两点y坐标相差不会太小 TODO: 阈值
                # 4.两点面积相近
                # 5.两点高宽比不超过3
                if cent1[0] > im_center_x and cent2[0] > im_center_x and \
                    min(cent1[1], cent2[1]) < im_center_y < max(cent1[1], cent2[1]) and \
                        x_diff < im_w * 0.02 and im_h * 0.02 < y_diff < im_h * 0.3 and \
                        max(area1, area2) * 1.0 / min(area1, area2) < 5 and \
                            wh1 < 3 and wh2 < 3:

                    pt_ind_pairs.append([i, j, x_diff])
            else:
                # 1.两点都在左侧
                # 2.两点在图像中央区域
                # 3.两点x坐标相差不大，两点y坐标相差不会太小
                # 4.两点面积相近
                if cent1[0] < im_center_x and cent2[0] < im_center_x and \
                    im_h * 0.25 < cent1[1] < im_h * 0.75 and im_h * 0.25 < cent2[1] < im_h * 0.75 and \
                        x_diff < im_w * 0.02 and im_h * 0.02 < y_diff < im_h * 0.3 and \
                            max(area1, area2) * 1.0 / min(area1, area2) < 5 and \
                                wh1 < 3 and wh2 < 3:
                    pt_ind_pairs.append([i, j, x_diff])

    pt_score = 0.0
    pt_pair = []
    if len(pt_ind_pairs) > 0:
        if len(pt_ind_pairs) > 1:
            pt_score = 0.5
            print('[WARNING] More than one laser point pair detected.')
        else:
            pt_score = 1.0
        # 若有多个时，取最可能的一对
        pt_ind_pairs = sorted(pt_ind_pairs, key=lambda p: p[2])
        pt_ind_pair = pt_ind_pairs[0]
        pt_pair = [centroids[pt_ind_pair[0]], centroids[pt_ind_pair[1]]]
        pt_label_mat[(pt_label_mat != pt_ind_pair[0]) & (pt_label_mat != pt_ind_pair[1])] = 0
        pt_mask[pt_label_mat == 0] = 0

        pt_mask = pt_mask.astype(np.uint8)
        pt_mask[pt_mask > 0] = 255

    laser_mask = laser_mask.astype(np.uint8)
    laser_mask[laser_mask > 0] = 255

    return pt_pair, pt_mask, laser_mask, pt_score





# if __name__ == '__main__':
#     im_path = '../../data/tree/3512.jpg'
#     im = cv2.imread(im_path)
#     segment_laser_pts(im)

