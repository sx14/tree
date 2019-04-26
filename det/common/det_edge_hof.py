# coding: utf-8
import numpy as np
from det.trunk.seg_trunk_thr import segment_trunk
from det.tag.seg_tag import segment_tag
from util.show_image import *


def detect_contour(im, trunk_map):
    _, contours, _ = cv2.findContours(trunk_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_map = np.zeros(im.shape)

    for i, contour in enumerate(contours):
        cv2.drawContours(contour_map, contours, i, (0, 0, 255), 1)
    contour_map = contour_map[:, :, 2].astype(np.uint8)

    return contour_map


def extract_lines(contour_map):
    im_empty = np.zeros((contour_map.shape[0], contour_map[1], 3))
    lines = cv2.HoughLinesP(contour_map, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # rho, theta = line[0]    # 获取极值ρ长度和θ角度
        # a = np.cos(theta)   # 获取角度cos值
        # b = np.sin(theta)   # 获取角度sin值
        # x0 = a * rho    # 获取x轴值
        # y0 = b * rho    # 获取y轴值　　x0和y0是直线的中点
        # x1 = int(x0 + 1000 * (-b))  # 获取这条直线最大值点x1
        # y1 = int(y0 + 1000 * (a))   # 获取这条直线最大值点y1
        # x2 = int(x0 - 1000 * (-b))  # 获取这条直线最小值点x2　　
        # y2 = int(y0 - 1000 * (a))   # 获取这条直线最小值点y2　　其中*1000是内部规则
        cv2.line(im_empty, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 开始划线
    im_empty = im_empty.astype(np.uint8)
    return im_empty


if __name__ == '__main__':
    # trunk
    im_path = '../data/tree1/4543.jpg'
    im = cv2.imread(im_path)
    trunk_map = segment_trunk(im)
    trunk_contour_map = detect_contour(im, trunk_map)
    show_images([im, trunk_map, trunk_contour_map])

    # tag
    im_path = '../data/tag/1.jpg'
    im = cv2.imread(im_path)
    tag_map = segment_tag(im)
    tag_contour_map = detect_contour(im, tag_map)
    show_images([im, tag_map, tag_contour_map])