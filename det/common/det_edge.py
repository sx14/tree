# coding: utf-8
import math

import cv2
import numpy as np

from util.show_image import *
from det.common.geo_utils import euc_dis


def detect_contour(trunk_mask):
    contours, _ = cv2.findContours(trunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros((trunk_mask.shape[0], trunk_mask.shape[1], 3))

    for i, contour in enumerate(contours):
        cv2.drawContours(contour_mask, contours, i, (0, 0, 255), 1)
    contour_mask = contour_mask[:, :, 2].astype(np.uint8)

    return contour_mask


def extract_lines_hof(contour_mask):
    im_empty = np.zeros((contour_mask.shape[0], contour_mask.shape[1], 3))
    lines = cv2.HoughLinesP(contour_mask, 1.0, np.pi / 180, 5, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(im_empty, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 开始划线
    im_empty = im_empty.astype(np.uint8)
    return im_empty


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


