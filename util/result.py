# coding: utf-8


class _InfoEnum:
    def __init__(self):
        # self.LASER_DET_FAILED = 'Laser point detection failed.'
        # self.STAND_TOO_CLOSE = 'Stand too close to tree.'
        # self.TRUNK_EDGE_UNCLEAR = 'Trunk edges are not clear.'
        # self.IMAGE_NOT_EXIST = 'Image not exist.'
        # self.BAD_MANUAL_ANNO = 'Bad manual annotation.'
        # self.SUCCESS = 'Success.'

        self.CALIBRATOR_DET_FAILED = 'CALIBRATOR_DET_FAILED'
        self.TRUNK_TOO_THICK = 'TRUNK_TOO_THICK'
        self.TRUNK_EDGE_UNCLEAR = 'TRUNK_EDGE_UNCLEAR'
        self.IMAGE_NOT_EXIST = 'IMAGE_NOT_EXIST'
        self.BAD_MANUAL_ANNO = 'BAD_MANUAL_ANNO'
        self.SUCCESS = 'SUCCESS'


InfoEnum = _InfoEnum()


class Result:

    def get_result(self):
        return self._result

    def __str__(self):
        return self._result.__str__()

    def __init__(self):

        # keys
        # 基本结果
        self._WIDTH = 'width'
        self._CONF = 'conf'
        self._INFO = 'info'
        self._IMAGE_PATH = 'image_path'
        self._TIME_CONSUME = 'time'

        # 树干四角点
        self._LEFT_TOP = 'left_top'
        self._RIGHT_TOP = 'right_top'
        self._LEFT_BOTTOM = 'left_bottom'
        self._RIGHT_BOTTOM = 'right_bottom'

        # 激光点
        self._CALIBRATE_POINTS = 'calibrate_points'

        # key-value
        self._result = {
            self._IMAGE_PATH: None,
            self._WIDTH: -1,
            self._CONF: -1,
            self._INFO: 'Error is too large.',
            self._LEFT_TOP: None,
            self._RIGHT_TOP: None,
            self._LEFT_BOTTOM: None,
            self._RIGHT_BOTTOM: None,
            self._CALIBRATE_POINTS: [],
            self._TIME_CONSUME: -1
        }

    def set_image_path(self, image_path):
        self._result[self._IMAGE_PATH] = image_path

    def set_width(self, width):
        self._result[self._WIDTH] = width

    def set_conf(self, conf):
        self._result[self._CONF] = conf

    def set_info(self, info):
        self._result[self._INFO] = info

    def set_trunk_left_top(self, point):
        self._result[self._LEFT_TOP] = point

    def set_trunk_left_bottom(self, point):
        self._result[self._LEFT_BOTTOM] = point

    def set_trunk_right_top(self, point):
        self._result[self._RIGHT_TOP] = point

    def set_trunk_right_bottom(self, point):
        self._result[self._RIGHT_BOTTOM] = point

    def set_calibrate_points(self, points):
        self._result[self._CALIBRATE_POINTS] = points

    def set_time(self, sec):
        self._result[self._TIME_CONSUME] = sec
