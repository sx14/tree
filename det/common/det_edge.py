# coding: utf-8
# status: reviewed

import cv2
import numpy as np


def detect_contour(trunk_mask):
    """
    根据树干掩码获得树干轮廓掩码
    :param trunk_mask: 树干掩码矩阵[0/1]
    :return: 树干轮廓矩阵[0/1]
    """
    contours, _ = cv2.findContours(trunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros((trunk_mask.shape[0], trunk_mask.shape[1], 3))

    for i, contour in enumerate(contours):
        cv2.drawContours(contour_mask, contours, i, (0, 0, 1), 1)
    contour_mask = contour_mask[:, :, 2].astype(np.uint8)

    return contour_mask


