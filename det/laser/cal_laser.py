# coding: utf-8
import cv2
import numpy as np
from config import *
from util.show_image import *
from det.common.geo_utils import *

class Laser:

    def __init__(self, pt_pair, pt_mask, laser_mask):
        # pt_pair: [[x1,y1], [x2,y2]]
        assert len(pt_pair) == 2
        self.pt_pair = [[int(pt[0]), int(pt[1])] for pt in pt_pair]
        self.pt_mask = pt_mask
        self.laser_mask = laser_mask
        self.CAMERA_FOCAL_LEN = FOCAL_LENGTH
        self.POINT_DISTANCE = POINT_DISTANCE

    def _point_pair_pixel_dis(self):
        # 激光点像素距离
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]
        return euc_dis(pt1, pt2)

    def RP_ratio(self):
        pixel_dis = self._point_pair_pixel_dis()
        read_dis = self.POINT_DISTANCE
        return read_dis / pixel_dis

    def shot_distance(self):
        ratio = self.RP_ratio()
        focal_len = self.CAMERA_FOCAL_LEN * 1.0
        if ratio is not None:
            return focal_len * ratio
        else:
            return None

    def cover_laser0(self, im):
        # DESPERATE 用纯色覆盖
        kernel = np.ones((3, 3), np.uint8)

        center_x = (self.pt_pair[0][0] + self.pt_pair[1][0]) / 2.0
        center_y = (self.pt_pair[0][1] + self.pt_pair[1][1]) / 2.0
        pt_dis = self._point_pair_pixel_dis()

        region_between_pts = im[int(center_x-pt_dis/4): int(center_x+pt_dis/4),
                                int(center_y-pt_dis/4): int(center_y+pt_dis/4), :]
        tree_color_b = np.mean(region_between_pts[:, :, 0])
        tree_color_g = np.mean(region_between_pts[:, :, 1])
        tree_color_r = np.mean(region_between_pts[:, :, 2])

        # 仅膨胀
        red_mask = self.laser_mask.copy()
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        im[red_mask > 0, 0] = tree_color_b
        im[red_mask > 0, 1] = tree_color_g
        im[red_mask > 0, 2] = tree_color_r

    def cover_laser(self, im):
        # 用图像块覆盖
        im_copy = im.copy()
        im_h, im_w, _ = im.shape

        laser_mask = self.laser_mask

        # laser_mask1 = laser_mask.copy()
        # laser_mask1[laser_mask1 > 0] = 255
        # show_image(laser_mask1)

        _, laser_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask.astype(np.uint8))

        # 找补丁中心位置
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]
        center_x = int((pt1[0] + pt2[0]) / 2.0)
        center_y = int((pt1[1] + pt2[1]) / 2.0)

        # 覆盖两点和线
        pts = [pt1, pt2, [center_x, center_y]]

        # 线可能不在正中，搜索多点
        line_search_step = 10
        ymin = min(pt1[1], pt2[1]) + line_search_step
        ymax = max(pt1[1], pt2[1]) - line_search_step
        pts_near_center = [[center_x, y] for y in range(int(ymin), int(ymax), 10)]
        pts += pts_near_center

        covered_labels = set()
        for i, pt in enumerate(pts):
            if i == 3:
                print('[WARNING]Laser Line is not in the middle of laser point pair.')

            laser_label = laser_label_mat[int(pt[1]), int(pt[0])]
            box_x1, box_y1, box_w, box_h, box_area = stats[laser_label]    # x,y,w,h,area
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h
            if laser_label == 0 or laser_label in covered_labels:
                continue

            covered_labels.add(laser_label)
            # 使用下方的图像块
            patch_x1 = box_x1
            patch_y1 = box_y1 + box_h * 2
            patch_x2 = box_x2
            patch_y2 = box_y2 + box_h * 2
            im_copy[box_y1: box_y2, box_x1: box_x2, :] = im_copy[patch_y1: patch_y2, patch_x1: patch_x2, :]
            if len(covered_labels) == 3:
                # 找到线
                break

        return im_copy


    def crop_image(self, im, n_dis_w=3, n_dis_h=3):
        im_h, im_w, _ = im.shape
        pt_dis = self._point_pair_pixel_dis()

        crop_center_x = (self.pt_pair[0][0] + self.pt_pair[1][0]) / 2.0
        crop_center_y = (self.pt_pair[0][1] + self.pt_pair[1][1]) / 2.0
        crop_w = pt_dis * n_dis_w
        crop_h = pt_dis * n_dis_h
        # default:
        # crop width:  3 * pt_dis
        # crop height: 3 * pt_dis
        crop_ymin = int(np.maximum(crop_center_y - crop_h / 2, 0))
        crop_ymax = int(np.minimum(crop_center_y + crop_h / 2, im_h))
        crop_xmin = int(np.maximum(crop_center_x - crop_w / 2, 0))
        crop_xmax = int(np.minimum(crop_center_x + crop_w / 2, im_w))

        im_patch = im[crop_ymin: crop_ymax + 1, crop_xmin: crop_xmax + 1, :]

        return im_patch

    def positive_pts(self):
        pt_dis = self._point_pair_pixel_dis()
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]
        if pt1[1] < pt2[1]:
            pt_up = pt1
            pt_dn = pt2
        else:
            pt_up = pt2
            pt_dn = pt1

        pos_pt1 = [int(pt_up[0]), int(max(pt_up[1] - pt_dis/2, 0))]
        pos_pt2 = [int(pt_dn[0]), int(max(pt_dn[1] + pt_dis/2, 0))]
        pos_pt3 = [int((pt_up[0]+pt_dn[0])/2), int((pt_up[1]+pt_dn[1])/2)]
        return [pos_pt1, pos_pt2, pos_pt3]