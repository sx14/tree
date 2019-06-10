# coding: utf-8
import numpy as np
import cv2

from util.show_image import *
from config import *


def clean_image(im):
    """
    只保留图像中央偏左部分区域
    用于激光点检测
    :param im:
    :return:
    """
    im_h, im_w, _ = im.shape
    mask = np.zeros(im.shape)
    # 0000
    # 0100
    # 0100
    # 0000
    mask[int(im_h * 0.25): int(im_h * 0.75), int(im_w * 0.25): int(im_w * 0.5), :] = 1
    cleaned_im = (im * mask).astype(np.uint8)
    return cleaned_im


def get_bright_pt_mask(im, laser_mask, show=False):
    # 激光点区域
    # 排除激光线

    # 膨胀红色区域，使其能够覆盖高亮点
    kernel = np.ones((3, 3), np.uint8)
    laser_mask_d = cv2.dilate(laser_mask, kernel, iterations=5)

    bright_mask = get_bright_mask(im, False)
    pt_mask = bright_mask & laser_mask_d

    # 膨胀高亮点，使其连成区域
    kernel = np.ones((3, 3), np.uint8)
    pt_mask = cv2.dilate(pt_mask.astype(np.uint8), kernel)

    if show:
        im_pt = pt_mask.astype(np.uint8)
        im_pt[im_pt > 0] = 255
        show_image(im_pt)

    return pt_mask


def get_bright_mask(im, show=False):
    """
    获得高亮区域二值掩码
    :param im:      原图
    :param show:    是否显示
    :return:
    """
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bright_mask = im_grey > 220

    # 1.膨胀高亮点，使其连成区域
    # 2.开运算，去毛边
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.dilate(bright_mask.astype(np.uint8), kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    if show:
        im_bright = bright_mask.astype(np.uint8)
        im_bright[im_bright > 0] = 255
        show_images([im_bright])

    return bright_mask


def get_laser_mask(im, dilate=False, show=False, use_hsv=True):
    """
    获得激光区域二值掩码
    :param im:      原图
    :param dilate:  是否仅膨胀
    :param show:    是否显示
    :return:        红色区域掩码
    """

    if use_hsv:
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(im_hsv, np.array([158,50,190]), np.array([180,255,255])).astype(np.int32)
        red2 = cv2.inRange(im_hsv, np.array([0,50,190]), np.array([6,255,255])).astype(np.int32)
        laser_mask = red1 + red2
        laser_mask[laser_mask > 0] = 1
        laser_mask = laser_mask.astype(np.uint8)
    else:
        im = im.astype(np.int32)
        laser_mask = im[:, :, 2] - np.max(im[:, :, [0, 1]], axis=2)
        laser_mask = laser_mask > 50
        laser_mask = laser_mask.astype(np.uint8)

    # 操作核
    kernel = np.ones((3, 3), np.uint8)
    if dilate:
        # 仅膨胀，红色激光区域彻底填补中央亮斑
        laser_mask = cv2.dilate(laser_mask, kernel, iterations=5)
    else:
        # 闭运算，去空洞
        laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
        # 开运算，去毛边
        laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    if show:
        im_red = laser_mask.astype(np.uint8)
        im_red[im_red > 0] = 255
        show_image(im_red)

    return laser_mask


def get_laser_points(im):
    """
    获取激光点坐标
    :param im: 原图
    :return:
        pt_pair:    激光点对坐标
        pt_mask:    激光点255掩码
        laser_mask: 激光255掩码
        pt_score:   激光点置信度
    """
    # 只保留中心区域
    cleaned_im = clean_image(im)
    # 尝试使用高亮区域寻找激光点
    pt_pair, pt_mask, laser_mask, pt_score = segment_laser_points(cleaned_im, True)
    if len(pt_pair) > 0:
        # 成功
        print('with bright.')
    else:
        # 高亮区域查找激光点失败
        # 降级，仅使用红色激光区域
        print('without bright.')
        pt_pair, pt_mask, laser_mask, pt_score = segment_laser_points(cleaned_im, False)

    # TODO: 没有激光线后，注释掉下一行
    laser_mask = get_laser_mask(cleaned_im, dilate=False, use_hsv=False)
    laser_mask[laser_mask > 0] = 255
    return pt_pair, pt_mask, laser_mask, pt_score


def segment_laser_points(im, use_bright=False):
    """
    分割激光点
    :param im:          清理过的图像
    :param use_bright:  是否仅使用高亮区域作为激光点
    :return:
        pt_pair:    激光点对坐标
        pt_mask:    激光点255掩码
        laser_mask: 激光255掩码
        pt_score:   激光点置信度
    """

    if use_bright:
        # 高亮区域与红色区域共同约束
        bright_mask = get_bright_mask(im, False)
        red_mask = get_laser_mask(im, dilate=True)
        laser_mask = bright_mask * red_mask
    else:
        # 仅使用红色区域
        laser_mask = get_laser_mask(im, dilate=False)

    _, pt_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask)
    pt_mask = laser_mask.copy()

    # 找两个激光点
    pt_ind_pairs = []
    im_h, im_w = pt_mask.shape
    im_center_x = im_w / 2.0
    im_center_y = im_h / 2.0
    for i in range(1, len(centroids)-1):
        for j in range(i+1, len(centroids)):

            # 两个区域的质心坐标
            center1_x, center1_y = centroids[i]
            center2_x, center2_y = centroids[j]

            # 两个区域的面积
            area1 = stats[i][-1]
            area2 = stats[j][-1]

            # 两个区域的box信息
            w1, h1 = stats[i][2:4]
            w2, h2 = stats[j][2:4]
            wh1 = max(w1, h1) * 1.0 / min(w1, h1)
            wh2 = max(w2, h2) * 1.0 / min(w2, h2)
            a1a2 = max(area1, area2) * 1.0 / min(area1, area2)

            # 两点水平方向距离
            x_diff = abs(center1_x - center2_x)
            # 两点竖直方向距离
            y_diff = abs(center1_y - center2_y)

            # 1.两点都在左侧
            # 2.两点在图像中央区域
            # 3.两点x坐标相差不大，两点y坐标相差不会太小
            # 4.两点面积相近
            if center1_x < im_center_x and \
                center2_x < im_center_x and \
                    im_h * 0.25 < center1_y < im_h * 0.75 and \
                    im_h * 0.25 < center2_y < im_h * 0.75 and \
                    x_diff < im_w * 0.02 and \
                    im_h * 0.02 < y_diff < im_h * 0.3 and \
                    a1a2 < 3 and wh1 < 2 and wh2 < 2:

                pt_ind_pairs.append([i, j, x_diff + wh1 + wh2 + a1a2])

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

        # 清除除两激光点以外的区域
        pt_label_mat[(pt_label_mat != pt_ind_pair[0]) & (pt_label_mat != pt_ind_pair[1])] = 0
        pt_mask[pt_label_mat == 0] = 0

        pt_mask = pt_mask.astype(np.uint8)
        pt_mask[pt_mask > 0] = 255

    laser_mask = laser_mask.astype(np.uint8)
    laser_mask[laser_mask > 0] = 255

    return pt_pair, pt_mask, laser_mask, pt_score

