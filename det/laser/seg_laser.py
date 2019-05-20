# coding: utf-8
import numpy as np
import cv2

from util.show_image import *


def get_pt_mask(im, laser_mask, show=False):
    # binary mat
    # 激光点区域
    # 排除激光线

    kernel = np.ones((3, 3), np.uint8)
    laser_mask_m = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
    laser_mask_d = cv2.dilate(laser_mask_m, kernel, iterations=10)

    bright_mask = get_bright(im)
    pt_mask = bright_mask & laser_mask_d
    pt_mask = cv2.morphologyEx(pt_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
    pt_mask = cv2.dilate(pt_mask.astype(np.uint8), kernel, iterations=2)

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

    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)

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

    if show:
        im_red = laser_mask.astype(np.uint8)
        im_red[im_red > 0] = 255
        show_image(im_red)

    return laser_mask


def segment_laser_pts(im):

    laser_mask = get_laser_mask(im, False)
    pt_mask = get_pt_mask(im, laser_mask, False)
    laser_mask = laser_mask | pt_mask

    # 所有高亮红色区域
    # centroid: [x,y]
    _, pt_label_mat, stats, centroids = cv2.connectedComponentsWithStats(pt_mask)

    # 找两个激光点
    pt_ind_pairs = []
    im_h, im_w = pt_mask.shape
    center_x = im_w / 2.0
    center_y = im_h / 2.0
    for i in range(1, len(centroids)-1):
        for j in range(i+1, len(centroids)):
            cent1 = centroids[i]
            cent2 = centroids[j]
            area1 = min(stats[i, 4], stats[j, 4])
            area2 = max(stats[i, 4], stats[j, 4])
            # 水平方向距离
            x_diff = abs(cent1[0] - cent2[0])
            # 竖直方向距离
            y_diff = abs(cent1[1] - cent2[1])
            # 与水平中线的距离差
            dis_diff = abs(cent1[1] + cent2[1] - im_h)

            # 1.两点都在右侧
            # 2.两点在中线一上一下
            # 3.两点x坐标相差不大
            # 4.两点y坐标相差不会太小
            # 5.两点面积相差应该不大 (x)
            if cent1[0] > center_x and cent2[0] > center_x and \
                min(cent1[1], cent2[1]) < center_y and \
                    max(cent1[1], cent2[1]) > center_y and \
                    x_diff < 40 and y_diff > 100:
                pt_ind_pairs.append([i, j, x_diff])

    pt_pair = []
    if len(pt_ind_pairs) > 0:
        if len(pt_ind_pairs) > 1:
            print('[WARNING] More than one laser point pair detected.')
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

    return pt_pair, pt_mask, laser_mask





# if __name__ == '__main__':
#     im_path = '../../data/tree/3512.jpg'
#     im = cv2.imread(im_path)
#     segment_laser_pts(im)

