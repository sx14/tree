# coding: utf-8
import numpy as np
import cv2


CALIBRATE_MODE = 0  # 标定模式
MEASURE_MODE = 1    # 测量模式


def get_bright_mask(im):
    """
    获取图像中的高亮度区域
    :param im: 原图
    :return:高亮区域掩码矩阵
            二值矩阵，高亮位置为1,其余为0
            与原图高宽相同
    TODO:   1.找更鲁棒的方法代替参数过滤
            2.在实际环境下测试后，可能需要调整
    """
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bright_mask = im_grey > 220

    kernel = np.ones((3, 3), np.uint8)
    # 闭运算，去空洞
    bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=5)
    # 开运算，去毛边
    bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=5)
    return bright_mask


def get_laser_mask(im):
    """
    获取图像中的红色激光区域
    :param im: 原图
    :return:激光区域掩码矩阵
            二值矩阵，红色位置为1,其余为0
            与原图高宽相同
    TODO:   1.找更鲁棒的方法代替参数过滤
            2.在实际环境下测试后，可能需要调整
    """
    im = im.astype(np.int32)    # BGR
    # R通道大于B通道大于G通道的区域
    laser_mask = im[:, :, 2] - np.max(im[:, :, [0, 1]], axis=2)
    laser_mask = laser_mask > 50
    laser_mask = laser_mask.astype(np.uint8)

    # 闭运算，去空洞
    kernel = np.ones((3, 3), np.uint8)
    laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=5)
    # 开运算，去毛边
    laser_mask = cv2.morphologyEx(laser_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=5)
    return laser_mask


def segment_laser_points(im, mode=False):
    """
    获取激光点对的坐标
    :param im:
    :param mode: CALIBRATE_MODE or MEASURE_MODE
    :return:
        pt_pair: 激光点对坐标，list，正常时len(list)=2
        pt_conf: 置信度
        pt_mask: 激光点掩码黑白图，用于debug
        laser_mask: 原始激光掩码黑白图，用于debug
    """

    if mode == CALIBRATE_MODE:
        # 标定时使用高亮点
        laser_mask = get_bright_mask(im)
    else:
        # 测量时使用激光点
        laser_mask = get_laser_mask(im)

    # 计算连通区域
    # pt_label_mat: 与图像等尺寸的整数矩阵，相同数字代表同一个连通区域
    # stats:        每个连通区域的外接框，每一行代表一个区域，[x,y,w,h,area]
    # centroids:    每个连通区域的质心坐标
    _, pt_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask)
    pt_mask = laser_mask.copy()

    # 找激光点对
    pt_ind_pairs = []   # 每个点：[index1, index2, x_diff]
    im_h, im_w = pt_mask.shape
    im_center_x = im_w / 2.0
    im_center_y = im_h / 2.0
    for i in range(1, len(centroids)-1):
        for j in range(i+1, len(centroids)):
            cent1 = centroids[i]
            cent2 = centroids[j]

            area1 = stats[i][-1]
            area2 = stats[j][-1]

            w1, h1 = stats[i][2:4]
            w2, h2 = stats[j][2:4]
            wh1 = max(w1, h1) * 1.0 / min(w1, h1)
            wh2 = max(w2, h2) * 1.0 / min(w2, h2)

            # 两点水平方向距离
            x_diff = abs(cent1[0] - cent2[0])
            # 两点竖直方向距离
            y_diff = abs(cent1[1] - cent2[1])

            # 1.两点都在左侧（激光器在镜头左侧）
            # 2.两点在图像中央区域（1/4~3/4）
            # 3.两点x坐标相差不大（理论上x_diff=0），两点y坐标相差不会太小（激光点距离10cm以上）
            # 4.两点面积比应接近1
            # 5.两点高宽比应接近1
            # TODO: 参数很敏感，需要改进
            if cent1[0] < im_center_x and cent2[0] < im_center_x and \
                im_h * 0.25 < cent1[1] < im_h * 0.75 and im_h * 0.25 < cent2[1] < im_h * 0.75 and \
                    x_diff < im_w * 0.02 and im_h * 0.02 < y_diff < im_h * 0.3 and \
                        max(area1, area2) * 1.0 / min(area1, area2) < 3 and \
                            wh1 < 3 and wh2 < 3:
                pt_ind_pairs.append([i, j, x_diff])

    pt_conf = 0.0  # 置信度
    pt_pair = []    # 激光点对
    if len(pt_ind_pairs) > 0:
        if len(pt_ind_pairs) > 1:
            # 多对候选
            # 置信度减半
            pt_conf = 0.5
            print('[WARNING] More than one point-pairs detected.')
        else:
            # 一对候选
            pt_conf = 1.0

        # 若有多个时，按x_diff排序，取x_diff最小的一对
        pt_ind_pairs = sorted(pt_ind_pairs, key=lambda p: p[2])
        pt_ind_pair = pt_ind_pairs[0]
        pt_pair = [centroids[pt_ind_pair[0]], centroids[pt_ind_pair[1]]]

        # 消除激光掩码中，非激光点对的红色区域
        pt_label_mat[(pt_label_mat != pt_ind_pair[0]) & (pt_label_mat != pt_ind_pair[1])] = 0
        pt_mask[pt_label_mat == 0] = 0

        # 转为灰度图[0-255]
        pt_mask = pt_mask.astype(np.uint8)
        pt_mask[pt_mask > 0] = 255

    # 原始激光掩码转为灰度图[0-255]
    laser_mask = laser_mask.astype(np.uint8)
    laser_mask[laser_mask > 0] = 255

    return pt_pair, pt_conf, pt_mask, laser_mask,

