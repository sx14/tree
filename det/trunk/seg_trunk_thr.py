# coding: utf-8

import cv2
import numpy as np
from util.show_image import *


def segment_trunk(im):
    """
    segment dominant tree trunk
    :param im: image(BGR)
    :return: binary mask. 1 indicates trunk, 0 indicates background.
    """
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # grey2bin, 1 is dark, 0 is bright
    thr, im_bin = cv2.threshold(im_grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 对图像进行“开运算”
    # 先腐蚀
    kernel = np.ones((3, 3), np.uint8)
    im_bin1 = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel, iterations=2)
    # 再膨胀
    im_bin2 = cv2.dilate(im_bin1, kernel, iterations=2)

    # 连通区域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(im_bin2)

    # show_images([im_grey, im_bin, im_bin1, im_bin2])

    # 假设图像中心在目标上
    # 找到图像中心所在的连通区域
    im_h, im_w = im_bin.shape
    im_ct_y = int(im_h / 2.0)
    im_ct_x = int(im_w / 2.0)
    dom_comp_label = labels[im_ct_y, im_ct_x]
    assert dom_comp_label > 0
    labels[labels != dom_comp_label] = 0
    labels[labels > 0] = 255
    labels = labels.astype(np.uint8)

    # visualize tree trunk
    # show_image(labels)

    return labels


if __name__ == '__main__':
    # test
    im_path = '../data/tree1/3513.jpg'
    im = cv2.imread(im_path)
    segment_trunk(im)