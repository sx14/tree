# coding: utf-8
# status: reviewed
import numpy as np
import cv2

from util.show_image import *
from config import *


def clean_image(im):
    """
    只保留图像中央偏左部分区域
    用于激光点检测
    # 0000
    # 0100
    # 0100
    # 0000
    :param im:
    :return:
    """
    im_h, im_w, _ = im.shape
    mask = np.zeros(im.shape)
    mask[int(im_h * 0.25): int(im_h * 0.75), int(im_w * 0.25): int(im_w * 0.5), :] = 1
    cleaned_im = (im * mask).astype(np.uint8)
    return cleaned_im


def get_bright_mask(im, show=False):
    """
    获得高亮区域二值掩码
    :param im:      原图
    :param show:    是否显示
    :return:        高亮区域01掩码
    """
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bright_mask = im_grey > 220

    # 1.膨胀高亮点，使其连成区域
    # 2.开运算，使边缘平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))
    bright_mask = cv2.dilate(bright_mask.astype(np.uint8), kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    if show:
        show_masked_image(im, bright_mask)
    bright_mask[bright_mask > 0] = 1
    return bright_mask


def get_laser_mask(im, dilate=False, show=False, use_hsv=True):
    """
    获得激光区域二值掩码
    :param im:      原图
    :param dilate:  是否仅膨胀
    :param show:    是否显示
    :param use_hsv: 是否使用HSV
    :return:        红色区域01掩码
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))
    if dilate:
        # 仅膨胀，红色激光区域彻底填补中央亮斑
        laser_mask = cv2.dilate(laser_mask, kernel, iterations=5)
    else:
        # 闭运算，开运算
        laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
        laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    if show:
        show_masked_image(im, laser_mask)
    laser_mask[laser_mask > 0] = 1
    return laser_mask


def get_laser_points(im, show=False):
    """
    获取激光点坐标
    :param im:      原图
    :param show:    是否可视化中间结果（debug only）
    :return:
        pt_pair:    激光点对坐标
        pt_mask:    激光点01掩码
        laser_mask: 激光01掩码
        pt_score:   激光点置信度

    """
    # 只保留中心区域
    cleaned_im = clean_image(im)
    # 尝试使用高亮区域寻找激光点
    pt_pair, pt_mask, laser_mask, pt_score = segment_laser_points(cleaned_im, use_bright=True, show=show)
    if len(pt_pair) <= 0:
        # 高亮区域查找激光点失败
        # 降级，仅使用红色激光区域
        pt_pair, pt_mask, laser_mask, pt_score = segment_laser_points(cleaned_im, use_bright=False, show=show)
        pt_score *= 0.6

    # TODO: 没有激光线后，注释掉下两行
    laser_mask = get_laser_mask(cleaned_im, dilate=False, use_hsv=False, show=False)
    # ===========================
    return pt_pair, pt_mask, laser_mask, pt_score


def segment_laser_points(im, use_bright=False, show=False):
    """
    分割激光点
    :param im:          清理过的图像
    :param use_bright:  是否仅使用高亮区域作为激光点
    :param show:        是否可视化中间结果（debug only）
    :return:
        pt_pair:    激光点对坐标
        pt_mask:    激光点255掩码
        laser_mask: 激光255掩码
        pt_score:   激光点置信度
    """

    if use_bright:
        # 高亮区域与红色区域共同约束
        bright_mask = get_bright_mask(im, show=show)
        red_mask = get_laser_mask(im, dilate=True, show=show)
        laser_mask = bright_mask * red_mask
    else:
        # 仅使用红色区域
        laser_mask = get_laser_mask(im, dilate=False, show=show)
    laser_mask[laser_mask > 0] = 1

    if show:
        visualize_image(laser_mask, 'show')

    # 找激光点对
    _, pt_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask)
    pt_mask = laser_mask.copy()

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
            # 3.两点x坐标相差不大
            # 4.两点y坐标相差不会太小
            # 5.两点面积相近
            # 6.两点高宽比例不会应该接近1
            if center1_x < im_center_x and \
                center2_x < im_center_x and \
                    im_h * 0.25 < center1_y < im_h * 0.75 and \
                    im_h * 0.25 < center2_y < im_h * 0.75 and \
                    x_diff < im_w * 0.02 and \
                    im_h * 0.06 < y_diff < im_h * 0.3 and \
                    a1a2 < 3 and wh1 < 2 and wh2 < 2:

                pt_ind_pairs.append([i, j, x_diff + wh1 + wh2 + a1a2])

    pt_score = 0.0
    pt_pair = []
    if len(pt_ind_pairs) > 0:
        if len(pt_ind_pairs) > 1:
            pt_score = 0.5
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
        pt_mask[pt_mask > 0] = 1

    return pt_pair, pt_mask, laser_mask, pt_score

