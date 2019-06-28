# coding: utf-8
# status: reviewed

import cv2
import numpy as np
from geo_utils import euc_dis
from util.show_image import show_images

def detect_contour(trunk_mask):
    """
    根据树干掩码获得树干轮廓掩码
    :param trunk_mask: 树干掩码矩阵[0/1]
    :return: 树干轮廓矩阵[0/1]
    """
    _, contours, _ = cv2.findContours(trunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros((trunk_mask.shape[0], trunk_mask.shape[1], 3))

    cv2.drawContours(contour_mask, contours, -1, (0, 0, 1), 1)
    contour_mask = contour_mask[:, :, 2].astype(np.uint8)
    # show_images([contour_mask])
    return contour_mask


def extract_lines_lsd(contour_mask):
    lsd = cv2.createLineSegmentDetector()
    contour_mask_copy = contour_mask.copy()
    contour_mask_copy[contour_mask_copy > 0] = 255
    lines = lsd.detect(contour_mask_copy)[0]
    lens = [euc_dis(line[0, :2].tolist(), line[0, 2:].tolist()) for line in lines]
    lens = np.array(lens)

    # remove extreme short lines.
    lines = lines[lens > 5]
    return lines

