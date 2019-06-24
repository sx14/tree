# coding: utf-8
from config import *
from det.common.geo_utils import *


class Calibrator:

    def __init__(self, calibrator_mask, conf):
        # pt_pair: [[x1,y1], [x2,y2]]
        self.calibrator_mask = calibrator_mask
        self.CAMERA_FOCAL_LEN = FOCAL_LENGTH
        self.POINT_DISTANCE = POINT_DISTANCE
        self.crop_box = None
        self.conf = conf

    def RP_ratio(self):
        raise NotImplementedError('not implemented')

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

    def get_conf(self):
        return self.conf

    def get_calibrate_points(self):
        raise NotImplementedError('not implemented')