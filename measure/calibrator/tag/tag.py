# coding: utf-8

from util.show_image import show_images

from measure.common.det_edge import *
from measure.common.geo_utils import *
from measure.calibrator.calibrator import Calibrator
from config import *


class BlueTag(Calibrator):

    def _return_flags(self):
        self.SUCCESS = 0
        self.EDGE_DET_FAILED = 1
        self.NOT_PARALLEL = 2
        self.NOT_PERPENDICULAR = 3

    def __init__(self, tag_mask):
        Calibrator.__init__(self, tag_mask, 1.0)
        self.WIDTH = TAG_WIDTH
        self.HEIGHT = TAG_HEIGHT
        self.FOCAL_LEN = FOCAL_LENGTH
        self._return_flags()

        self.mask = tag_mask
        raw_lines = extract_lines_lsd(tag_mask)
        # show_images([line_mask])
        self.edges = []
        for i, line in enumerate(raw_lines):
            self.edges.append(Edge(line[0, :2], line[0, 2:]))
        self._connect_lines()

        # debug, show lines
        # im_show = np.zeros((self.mask.shape[0], self.mask.shape[1], 3)).astype(np.uint8)
        # for i, line in enumerate(raw_lines):
        #     x1,y1,x2,y2 = line[0]
        #     cv2.line(im_show, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # show_images([im_show])

    def _connect_lines(self):
        # 连接中断的直线段

        MIN_PARALLEL_ANGLE = 5      # 角度制
        MIN_CONNECT_DISTANCE = 5    # 欧氏距离

        # 标记可连接处
        for i in range(len(self.edges)):
            for j in range(i+1, len(self.edges)):
                # 取两条不同的直线段
                l1 = self.edges[i]
                l2 = self.edges[j]

                # 用向量表示两条直线的方向
                v1 = l1.vec()
                v2 = l2.vec()

                # 计算夹角（角度制）
                alpha = angle(v1, v2)
                # 夹角小于5'时，认为是同一条直线
                if alpha < MIN_PARALLEL_ANGLE:
                    # 应该是同一条直线，检查是否首尾相接
                    # 尝试每一对顶点
                    for p in range(2):
                        for q in range(2):
                            l1_pt = l1.get_pt(p)
                            l2_pt = l2.get_pt(q)
                            dis = euc_dis(l1_pt, l2_pt)
                            if dis < MIN_CONNECT_DISTANCE:
                                # 若两点距离小于5像素，则认为是相接的，可以连接
                                if l1.get_connect(p) is None and l2.get_connect(q) is None:
                                    # l1的第p个顶点 -- 连接 -- l2的第q个顶点
                                    l1.set_connect(p, l2, q)
                                    # l2的第q个顶点 -- 连接 -- l1的第p个顶点
                                    l2.set_connect(q, l1, p)
        # 连接
        further_connect = True
        while further_connect:
            # 每轮连接两条边
            # 若这一轮没有连接，则停止
            new_edges = []
            for i, l1 in enumerate(self.edges):
                connection1 = l1.get_connect(1)
                if connection1 is not None and not l1.merged and not connection1[0].merged:
                    l2 = connection1[0]
                    # 连接一对直线段，产生一个新的直线段，丢弃旧的两个
                    merged_edge = Edge(l1.get_pt(0), l2.get_pt(1))

                    # 两直线短的连接处(edge, pt_ind)
                    l1_connect = l1.get_connect(0)
                    l2_connect = l2.get_connect(1)

                    # 连接，维护连接点
                    # 1. 新线段继承旧线段的连接点
                    # 2. 与被合并线段连接的线段，更新连接点指向新线段
                    if l1_connect is not None:
                        # (连接的直线段，连接的端点)
                        l1_connect_l, l1_connect_pt_ind = l1_connect
                        l1_connect_l.set_connect(l1_connect_pt_ind, merged_edge, 0)
                        merged_edge.set_connect(0, l1_connect[0], l1_connect[1])
                    if l2_connect is not None:
                        l2_connect_l, l2_connect_pt_ind = l2_connect
                        l2_connect_l.set_connect(l2_connect_pt_ind, merged_edge, 1)
                        merged_edge.set_connect(1, l2_connect[0], l2_connect[1])

                    new_edges.append(merged_edge)
                    l1.merged = True
                    l2.merged = True

            for i, l in enumerate(self.edges):
                if not l.merged:
                    new_edges.append(l)

            if len(new_edges) == len(self.edges):
                # 这一轮没有连接，停止
                further_connect = False
            self.edges = new_edges

    def show_lines(self):
        sorted_edges = sorted(self.edges, key=lambda e: e.length(), reverse=True)
        im_empty = np.zeros((self.mask.shape[0], self.mask.shape[1], 3))
        for line in sorted_edges:
            x1 = line.pts[0][0]
            y1 = line.pts[0][1]
            x2 = line.pts[1][0]
            y2 = line.pts[1][1]
            cv2.line(im_empty, (x1, y1), (x2, y2), (0, 0, 255), 1)
        im_show = im_empty.astype(np.uint8)
        return im_show

    def is_edge_det_succ(self):
        return len(self.edges) >= 3

    def check_parallel(self):
        if not self.is_edge_det_succ():
            return self.EDGE_DET_FAILED
        # 按长度排序
        edges = sorted(self.edges, key=lambda e: e.length(), reverse=True)

        # 取最长边
        edge1 = edges[0]
        edge2 = edges[1]

        alpha = angle(edge1.vec(), edge2.vec())
        if alpha < 2:
            return self.SUCCESS
        else:
            return self.NOT_PARALLEL

    def check_perpendicular(self):
        if not self.is_edge_det_succ():
            return self.EDGE_DET_FAILED

        # 按长度排序
        edges = sorted(self.edges, key=lambda e: e.length(), reverse=True)

        # 取最长边
        edge1 = edges[0]
        edge2 = edges[1]

        # 找与edge1, edge2垂直的edge
        edge3 = None
        for edge in edges[2:]:
            alpha1 = angle(edge1.vec(), edge.vec())
            alpha2 = angle(edge2.vec(), edge.vec())
            if abs(180-alpha1-alpha2) < 10:
                edge3 = edge
                break

        if edge3 is not None:
            len_ratio = edge3.length() / edge1.length()
            angle_diff1 = abs(angle(edge1.vec(), edge3.vec()) - 90)
            angle_diff2 = abs(angle(edge2.vec(), edge3.vec()) - 90)
            angle_diff = max(angle_diff1, angle_diff2)
            if len_ratio < 0.1:
                return self.EDGE_DET_FAILED
            if angle_diff > 2:
                return self.NOT_PERPENDICULAR
            return self.SUCCESS
        else:
            return self.EDGE_DET_FAILED

    def pixel_width(self):
        # 按长度排序
        edges = sorted(self.edges, key=lambda e: e.length(), reverse=True)

        # 取最长边
        edge1 = edges[0]
        edge2 = edges[1]
        A1, B1, C1 = edge1.normal()
        A2, B2, C2 = edge2.normal()

        alpha = angle(edge1.vec(), edge2.vec())
        if alpha < 5:
            # 计算距离
            e1_pts = edge1.get_pts(5)
            e1_e2_dis = []
            for e1_pt in e1_pts:
                A1p,B1p,C1p = edge1.normal_perpendicular(e1_pt)
                e2_pt = cal_cross_pt([A2,B2,C2], [A1p,B1p,C1p])
                if e2_pt is not None:
                    e1_e2_dis.append(euc_dis(e1_pt, e2_pt))
            # 计算均值
            if len(e1_e2_dis) > 0:
                return sum(e1_e2_dis)*1.0/len(e1_e2_dis)
            else:
                return None
        else:
            return None

    def pixel_height(self):
        # 按长度排序
        edges = sorted(self.edges, key=lambda e: e.length(), reverse=True)

        # 取最长边
        edge1 = edges[0]
        edge2 = edges[1]

        alpha = angle(edge1.vec(), edge2.vec())
        if alpha < 5:
            # 计算距离
            # edge1 垂线
            p_edge1 = edge1.normal_perpendicular(edge1.get_pt(0))
            contour = detect_contour(self.mask)
            ys, xs = np.where(contour > 0)
            max_dis = float('-Inf')
            min_dis = float('+Inf')
            for i in range(len(xs)):
                dis = signed_pt_2_line(p_edge1, [xs[i], ys[i]])
                if dis is not None:
                    max_dis = max(max_dis, dis)
                    min_dis = min(min_dis, dis)
            return max_dis - min_dis
        else:
            return None

    def _RP_ratio_w(self):
        # Tag的实际宽度 / Tag的像素宽度
        pixel_width = self.pixel_width()
        real_width = self.WIDTH * 1.0
        if pixel_width is not None:
            return real_width / pixel_width
        else:
            return None

    def _RP_ratio_h(self):
        # Tag的实际高度 / Tag的像素高度
        pixel_height = self.pixel_height()
        real_height = self.HEIGHT * 1.0
        if pixel_height is not None:
            return real_height / pixel_height
        else:
            return None

    def RP_ratio(self):
        w_ratio = self._RP_ratio_w()
        h_ratio = self._RP_ratio_h()
        if w_ratio is None:
            return h_ratio
        elif h_ratio is None:
            return w_ratio
        else:
            return (w_ratio + h_ratio) / 2.0

    def shot_distance(self):
        ratio = self.RP_ratio()
        focal_len = self.FOCAL_LEN * 1.0
        if ratio is not None:
            return focal_len * ratio
        else:
            return None

    def get_calibrate_points(self):
        # 按长度排序
        edges = sorted(self.edges, key=lambda e: e.length(), reverse=True)

        # 取最长边
        edge1 = edges[0]
        edge2 = edges[1]

        # 保存4个标定点
        calibrate_points = []

        # 计算长边夹角
        alpha = angle(edge1.vec(), edge2.vec())
        if alpha < 5:

            # 计算长轴方向的2个标定点
            A1, B1, C1 = edge1.normal()
            A2, B2, C2 = edge2.normal()

            # 计算edge1和edge2与x轴交点的中点
            y_temp = 0
            x1_temp = (-C1) / A1
            x2_temp = (-C2) / A2
            pt_mid = ((x1_temp + x2_temp) / 2.0, y_temp)

            Am = A1
            Bm = B1
            Cm = (-A1 * pt_mid[0]) + (-B1 * pt_mid[1])

            long_cal_points = []
            contour = detect_contour(self.mask)
            ys, xs = np.where(contour > 0)
            for i in range(len(xs)):
                ptx = xs[i]
                pty = ys[i]
                diff = abs(Am * ptx + Bm + pty + Cm)
                if diff < 0.1:
                    long_cal_points.append([ptx, pty])

            if len(long_cal_points) >= 2:
                calibrate_points += long_cal_points

            # 计算短轴方向的2个标定点
            pt1 = edge1.get_pt(0)
            pt2 = edge1.get_pt(1)

            ptm = [(pt1[0]+pt2[0])/2.0, (pt1[1]+pt2[1])/2.0]
            edge1_p = edge1.normal_perpendicular(ptm)

            short_cal_points = []
            if edge1_p is not None:
                for i in range(len(xs)):
                    ptx = xs[i]
                    pty = ys[i]
                    if abs(Am * ptx + Bm + pty + Cm) < 0.1:
                        short_cal_points.append([ptx, pty])
            if len(short_cal_points) >= 2:
                calibrate_points += short_cal_points

        return calibrate_points

    def cover_calibrator(self, im):
        """
        用<蓝色标签>上方的树皮
        掩盖<蓝色标签>
        :param im: 图像(cropped)
        :param tag_map: 标签掩码
        :return: 去除标签后的图像
        """
        im_copy = im.copy()
        im_h, im_w, _ = im_copy.shape
        tag_map = self.calibrator_mask

        tag_ys, tag_xs = np.where(tag_map > 0)
        ymin = tag_ys.min() - 3
        ymax = tag_ys.max() + 3
        xmin = tag_xs.min() - 3
        xmax = tag_xs.max() + 3

        margin = int((ymax - ymin)/2)
        ymin_above = np.maximum(ymin - margin, 0)
        margin = ymin - ymin_above
        region_above_tag = im_copy[ymin_above:ymin, xmin:xmax, :]

        for i in range(ymin, ymax, margin):
            im_copy[i:i+margin, xmin:xmax] = region_above_tag

        return im_copy

    def crop_image(self, im, n_dis_w=4, n_dis_h=3):
        # default:
        # crop width:  4 * height
        # crop height: 3 * height
        # 中心为两激光点中心

        im_h, im_w, _ = im.shape
        pt_dis = self.pixel_height()

        tag_ys, tag_xs = np.where(self.calibrator_mask > 0)
        ymin = tag_ys.min()
        ymax = tag_ys.max()
        xmin = tag_xs.min()
        xmax = tag_xs.max()

        crop_center_x = (xmin + xmax) / 2.0
        crop_center_y = (ymin + ymax) / 2.0
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

        return im_patch

    def positive_pts(self):
        """
        标签上下各取一个点
        :return: 坐标列表
        """
        im_h, im_w = self.calibrator_mask.shape
        tag_ys, tag_xs = np.where(self.calibrator_mask > 0)
        ymin = tag_ys.min()
        ymax = tag_ys.max()
        xmin = tag_xs.min()
        xmax = tag_xs.max()

        if self.crop_box is not None:
            ymin = ymin - self.crop_box['ymin']
            xmin = xmin - self.crop_box['xmin']
            ymax = ymax - self.crop_box['ymin']
            xmax = xmax - self.crop_box['xmin']

        pt1_x = int((xmin + xmax) / 2)
        pt1_y = max(ymin - 50, 0)
        pt2_x = pt1_x
        pt2_y = min(ymax + 50, im_h-1)

        return [[pt1_x, pt1_y], [pt2_x, pt2_y]]


