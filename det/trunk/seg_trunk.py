# coding: utf-8

import cv2
import numpy as np
from util.show_image import *
from Intseg.our_func_cvpr18 import our_func_sunx

def segment_leaf(im):
    im_h, im_w, _ = im.shape
    im = im.astype(np.int32)
    im_bin = im[:, :, 1] - np.max(im[:, :, [0, 2]], axis=2)
    leaf_mask = np.zeros(im.shape[:2]).astype(np.uint8)
    leaf_mask[im_bin > 10] = 255
    return leaf_mask


def segment_cloth(im):
    im_h, im_w, _ = im.shape
    im = im.astype(np.int32)
    im_bin = im[:, :, 0] - np.max(im[:, :, [1, 2]], axis=2)
    cloth_mask = np.zeros(im.shape[:2]).astype(np.uint8)
    cloth_mask[im_bin > 0] = 255
    return cloth_mask


def segment_trunk_thr(im, pr_bg_mask):
    """
    DEPRECATED: not robust
    segment dominant tree trunk
    :param im: image(BGR)
    :return: binary mask. 1 indicates trunk, 0 indicates background.
    """
    im[pr_bg_mask > 0, :] = 255
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # grey2bin, 1 is dark, 0 is bright
    thr, im_bin = cv2.threshold(im_grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 对图像开运算
    # 腐蚀
    kernel = np.ones((3, 3), np.uint8)
    im_bin1 = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel, iterations=10)
    # 膨胀
    im_bin2 = cv2.dilate(im_bin1, kernel, iterations=10)

    # 连通区域
    labels = get_center_connected_component(im_bin2)

    show_img = im * labels[:, :, np.newaxis]
    show_img[labels == 0, :] = 0

    labels[labels > 0] = 255
    labels = labels.astype(np.uint8)
    return show_img, labels


def segment_trunk_ff(im, tag_mask, pr_bg_mask):
    im_h, im_w, _ = im.shape
    im_copy = im.copy()
    mask = np.zeros((im_h+2, im_w+2)).astype(np.uint8)
    tag_points = np.where(tag_mask > 0)
    seed_y = int((np.max(tag_points[0]) + np.min(tag_points[0])) / 2.0)
    seed_x = int((np.max(tag_points[1]) + np.min(tag_points[1])) / 2.0)

    cv2.floodFill(im_copy, mask, (seed_y, seed_x), (0, 0, 255), (50, 50, 50), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)
    # show_image('', im_copy)
    return im_copy, mask


def segment_trunk(im, tag_mask, bg_mask):

    # 将图像中确定是背景的区域设为黑色
    # im[pr_bg_mask > 0, :] = 0
    im_h, im_w, _ = im.shape

    # 标签的box
    tag_ys, tag_xs = np.where(tag_mask > 0)
    ymin = tag_ys.min()
    ymax = tag_ys.max()
    xmin = tag_xs.min()
    xmax = tag_xs.max()
    tag_h = ymax - ymin
    tag_w = xmax - xmin

    # 准备mask for GrabCut
    mask = np.zeros(tag_mask.shape).astype(np.uint8)

    # 候选包围框：能够完整包含前景的区域（包含tag的一条）
    # w: 5 * tag_w
    pr_n_tag_w = 2
    pr_fg_xmin = np.maximum(xmin - int(pr_n_tag_w * tag_w), 0)
    pr_fg_xmax = np.minimum(xmax + int(pr_n_tag_w * tag_w), im_w)
    mask[:, pr_fg_xmin:pr_fg_xmax] = cv2.GC_PR_FGD

    # 标记background: bg region
    mask[bg_mask > 0] = cv2.GC_BGD
    _, pr_fg_mask = segment_trunk_thr(im, bg_mask)
    pr_fg_mask[pr_fg_mask > 0] = 1
    pr_bg_mask = (~pr_fg_mask.astype(np.bool))
    mask[(mask == cv2.GC_PR_FGD) & pr_bg_mask] = cv2.GC_PR_BGD

    # mask1 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    # mask1[mask1 > 0] = 255
    # show_images('', [im, mask1])

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(im, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    mask = get_center_connected_component(mask)
    show_img = im * mask[:, :, np.newaxis]
    show_img[mask == 0, :] = 0
    mask[mask > 0] = 255
    return show_img, mask


def get_center_connected_component(mask):
    # 连通区域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # 假设图像中心在目标上
    # 找到图像中心所在的连通区域
    im_h, im_w = mask.shape
    im_ct_y = int(im_h / 2.0)
    im_ct_x = int(im_w / 2.0)
    dom_comp_label = labels[im_ct_y, im_ct_x]
    if dom_comp_label > 0:
        labels[labels != dom_comp_label] = 0
    else:
        print('[WARNING] image center not fg.')
    labels[labels > 0] = 1
    return labels.astype(np.uint8)


def segment_trunk_int(im, tag_mask, pr_bg_mask, im_id=1, user_id =1):
    # positive points
    # 标签的box上下50像素
    tag_ys, tag_xs = np.where(tag_mask > 0)
    ymin = tag_ys.min()
    ymax = tag_ys.max()
    xmin = tag_xs.min()
    xmax = tag_xs.max()

    pos_y1 = ymin - 50
    pos_y2 = ymax + 50
    pos_x = int((xmin+xmax)/2.0)
    pos_pts = [[pos_y1, pos_x], [pos_y2, pos_x]]

    # negative points
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(pr_bg_mask)

    areas = stats[:, -1]
    sorted_inds = np.argsort(areas)[::-1]
    sorted_inds = sorted_inds[1:min(6, len(sorted_inds))]
    centroids = centroids[sorted_inds]

    neg_pts = centroids.astype(np.int32).tolist()

    pts = pos_pts + neg_pts
    pns = [1, 1] + [0 for _ in range(len(neg_pts))]

    mask = our_func_sunx(user_id, im_id, im, pns, pts)
    show_img = im * mask[:, :, np.newaxis]
    show_img[mask == 0, :] = 0
    mask[mask > 0] = 255
    return show_img, mask
