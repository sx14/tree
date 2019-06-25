# coding: utf-8
# status: reviewed

import cv2
import numpy as np
from geo_utils import euc_dis


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


def extract_lines_lsd(contour_mask):
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(contour_mask)[0]
    lens = [euc_dis(line[0, :2].tolist(), line[0, 2:].tolist()) for line in lines]
    lens = np.array(lens)

    # remove extreme short lines.
    # TODO: parameter
    lines = lines[lens > 5]

    # only for show
    im_show = np.zeros((contour_mask.shape[0], contour_mask.shape[1], 3))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(im_show, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # show_image('', im_show)
    im_show = im_show.astype(np.uint8)
    return im_show, lines

