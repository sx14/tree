# coding: utf-8
from config import *
from det.common.geo_utils import *


class Calibrator:

    def __init__(self, pt_pair, calibrator_mask):
        # pt_pair: [[x1,y1], [x2,y2]]
        assert len(pt_pair) == 2
        self.org_pt_pair = [[int(pt[0]), int(pt[1])] for pt in pt_pair]
        self.pt_pair = [[int(pt[0]), int(pt[1])] for pt in pt_pair]
        self.calibrator_mask = calibrator_mask
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
        # 标定点像素距离
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

    def cover_calibrator(self, im):
        raise NotImplementedError('not implemented')

    def crop_image(self, im, n_dis_w=3, n_dis_h=2):
        raise NotImplementedError('not implemented')

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
        raise NotImplementedError('not implemented')