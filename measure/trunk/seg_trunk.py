# coding: utf-8
import os
import shutil

import cv2
import numpy as np

from util.show_image import *
from Intseg.our_func_cvpr18 import our_func_sunx


def get_center_connected_component(mask):
    """
    获得位于中心位置的连通区域
    :param mask: 掩码矩阵
    :return: 仅保留中心位置连通区域的掩码矩阵
    """
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
    labels[labels > 0] = 1
    return labels.astype(np.uint8)


def segment_trunk_int(im, pos_pts, pr_bg_mask=None, im_id=0, user_id=0):

    # 8 negative points
    neg_pts = []
    if pr_bg_mask is not None:
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(pr_bg_mask)

        areas = stats[:, -1]
        large_labels = np.argsort(areas)[::-1]
        large_labels = large_labels[1:min(5, len(large_labels))]
        # 4 pts
        neg_pts = [[c[1], c[0]] for c in centroids[large_labels].astype(np.int32).tolist()]

    # 4 pts
    # 取四个角作为negative points
    margin = 5
    im_h, im_w, _ = im.shape
    neg_pts += [[margin, margin],
               [margin, im_w - margin],
               [im_h - margin, margin],
               [im_h - margin, im_w - margin]]
    pos_pts = [[pt[1], pt[0]] for pt in pos_pts]

    if os.path.isdir('Intseg/res'):
        shutil.rmtree('Intseg/res')

    pts = pos_pts + neg_pts
    pns = [1] * len(pos_pts) + [0] * len(neg_pts)
    mask = np.zeros(im.shape[:2]).astype(np.uint8)
    for i in range(len(pts)):
        mask = our_func_sunx(user_id, im_id, im, i, pns[i], pts[i])
    mask = get_center_connected_component(mask)
    mask[mask > 0] = 1
    return mask
