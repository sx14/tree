# coding: utf-8
import cv2
import copy
import numpy as np
from config import *
from util.show_image import *
from det.common.geo_utils import *


class Laser:

    def __init__(self, pt_pair, pt_mask, laser_mask):
        # pt_pair: [[x1,y1], [x2,y2]]
        assert len(pt_pair) == 2
        self.org_pt_pair = [[int(pt[0]), int(pt[1])] for pt in pt_pair]
        self.pt_pair = [[int(pt[0]), int(pt[1])] for pt in pt_pair]
        self.pt_mask = pt_mask
        self.laser_mask = laser_mask
        self.CAMERA_FOCAL_LEN = FOCAL_LENGTH
        self.POINT_DISTANCE = POINT_DISTANCE
        self.crop_box = None

    def get_top_pt(self):
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]

        if pt1[1] < pt2[1]:
            return pt1
        else:
            return pt2

    def get_bottom_pt(self):
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]

        if pt1[1] > pt2[1]:
            return pt1
        else:
            return pt2

    def point_pixel_dis(self):
        # 激光点像素距离
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]
        return euc_dis(pt1, pt2)

    def RP_ratio(self):
        pixel_dis = self.point_pixel_dis()
        read_dis = self.POINT_DISTANCE
        return read_dis / pixel_dis

    def shot_distance(self):
        ratio = self.RP_ratio()
        focal_len = self.CAMERA_FOCAL_LEN * 1.0
        if ratio is not None:
            return focal_len * ratio
        else:
            return None

    def cover_laser(self, im, has_laser_line=True):
        # 用图像块覆盖
        im_copy = im.copy()
        im_h, im_w, _ = im.shape

        laser_mask = self.laser_mask
        _, laser_label_mat, stats, centroids = cv2.connectedComponentsWithStats(laser_mask.astype(np.uint8))

        # 找激光点中心位置
        pt1 = sorted(self.pt_pair, key=lambda pt: pt[1])[0]
        pt2 = sorted(self.pt_pair, key=lambda pt: pt[1])[1]
        center_x = int((pt1[0] + pt2[0]) / 2.0)
        center_y = int((pt1[1] + pt2[1]) / 2.0)
        pts = [pt1]

        if has_laser_line:
            # 线在两点之间，搜索多点
            pt_center = [center_x, center_y]
            pts.append(pt_center)
            line_search_step = 2
            ymin = min(pt1[1], pt2[1]) + line_search_step
            ymax = max(pt1[1], pt2[1]) - line_search_step
            pts_near_center = [[center_x, y] for y in range(int(ymin), int(ymax), line_search_step)]
            pts += pts_near_center

        pts.append(pt2)

        covered_labels = set()
        for i, pt in enumerate(pts):
            laser_label = laser_label_mat[int(pt[1]), int(pt[0])]
            box_x1, box_y1, box_w, box_h, box_area = stats[laser_label]    # x,y,w,h,area
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h
            if laser_label == 0 or laser_label in covered_labels:
                # 不是激光区域，或已经找到的激光区域
                continue

            covered_labels.add(laser_label)
            # 使用上方的图像块覆盖
            patch_x1 = box_x1
            patch_y1 = box_y1 - box_h * 2
            patch_x2 = box_x2
            patch_y2 = box_y2 - box_h * 2
            im_copy[box_y1: box_y2, box_x1: box_x2, :] = im_copy[patch_y1: patch_y2, patch_x1: patch_x2, :]

        return im_copy

    def crop_image(self, im, n_dis_w=3, n_dis_h=2):
        # default:
        # crop width:  3 * pt_dis
        # crop height: 3 * pt_dis
        # 中心为两激光点中心

        im_h, im_w, _ = im.shape
        pt_dis = self.point_pixel_dis()

        # 恢复激光点坐标
        self.pt_pair = copy.deepcopy(self.org_pt_pair)

        crop_center_x = (self.pt_pair[0][0] + self.pt_pair[1][0]) / 2.0
        crop_center_y = (self.pt_pair[0][1] + self.pt_pair[1][1]) / 2.0
        crop_w = pt_dis * n_dis_w
        crop_h = pt_dis * n_dis_h

        crop_ymin = int(np.maximum(crop_center_y - crop_h / 2, 0))
        crop_ymax = int(np.minimum(crop_center_y + crop_h / 2, im_h))
        crop_xmin = int(np.maximum(crop_center_x - crop_w / 2, 0))
        crop_xmax = int(np.minimum(crop_center_x + crop_w / 2, im_w))

        self.crop_box = {
            'xmin': crop_xmin,
            'ymin': crop_ymin,
            'xmax': crop_xmax,
            'ymax': crop_ymax
        }

        im_patch = im[crop_ymin: crop_ymax + 1, crop_xmin: crop_xmax + 1, :]

        # 更新激光点的坐标
        # pt1 = self.pt_pair[0]
        # pt2 = self.pt_pair[1]
        # pt1[0] = pt1[0] - crop_xmin
        # pt1[1] = pt1[1] - crop_ymin
        # pt2[0] = pt2[0] - crop_xmin
        # pt2[1] = pt2[1] - crop_ymin

        return im_patch

    def recover_coordinate(self, pt):
        if self.crop_box is not None:
            crop_ymin = self.crop_box['ymin']
            crop_xmin = self.crop_box['xmin']
        else:
            crop_ymin = 0
            crop_xmin = 0

        org_x = pt[0] + crop_xmin
        org_y = pt[1] + crop_ymin
        return [org_x, org_y]

    def positive_pts(self):
        pt_dis = self.point_pixel_dis()
        pt1 = self.pt_pair[0]
        pt2 = self.pt_pair[1]

        if self.crop_box is not None:
            pt1[0] = pt1[0] - self.crop_box['xmin']
            pt1[1] = pt1[1] - self.crop_box['ymin']
            pt2[0] = pt2[0] - self.crop_box['xmin']
            pt2[1] = pt2[1] - self.crop_box['ymin']

        if pt1[1] < pt2[1]:
            pt_up = pt1
            pt_dn = pt2
        else:
            pt_up = pt2
            pt_dn = pt1

        pos_pt1 = [int(pt_up[0]), int(max(pt_up[1] - pt_dis/4, 0))]
        pos_pt2 = [int(pt_dn[0]), int(max(pt_dn[1] + pt_dis/4, 0))]
        pos_pt3 = [int((pt_up[0]+pt_dn[0])/2), int((pt_up[1]+pt_dn[1])/2)]
        return [pos_pt1, pos_pt2, pos_pt3]