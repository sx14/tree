# coding: utf-8
import cv2
from scipy.optimize import curve_fit

from measure.common.det_edge import *
from measure.common.geo_utils import *
from util.show_image import visualize_image


class Trunk:
    def __init__(self, trunk_mask):
        self.trunk_mask = trunk_mask
        self.contour_mask = detect_contour(trunk_mask)
        h, w = self.contour_mask.shape
        self.contour_mask = self.contour_mask[4:h - 4, :]

    @staticmethod
    def _get_edge_pts(sub_mask):
        ys, xs = np.where(sub_mask > 0)
        pts = [[xs[i], ys[i]] for i in range(len(xs))]
        if len(pts) == 0:
            return [], []
        # sorted by: y-first; x-second
        pts = sorted(pts, key=lambda a: (a[1], a[0]))
        l_pt_q = [pts.pop(0)]   # 左边缘上顶点
        l_pts = []
        PROTECT_COUNT = 30000
        cnt = 0
        while len(l_pt_q) > 0 and cnt < PROTECT_COUNT:
            cnt += 1
            curr = l_pt_q.pop(0)
            l_pts.append(curr)
            l_pt_inds = []
            for i, pt in enumerate(pts):
                if eight_connected(curr, pt):
                    l_pt_q.append(pt)
                    l_pt_inds.insert(0, i)
                elif pt[1]-curr[1] >= 2:
                    break
            # 从pts中移除属于左边缘的点
            l_pt_inds = sorted(l_pt_inds, reverse=True)
            for l_pt_ind in l_pt_inds:
                pts.pop(l_pt_ind)
        r_pts = pts
        return l_pts, r_pts

    @staticmethod
    def _width(normal_line, left_pts, right_pts):
        """
        .   .
        .   .
        .   .
        .----.
        .     .
        left_pts上应该存在一点A在normal_line上
        right_pts上应该存在一点B在normal_line上
        计算点A和B的欧式距离

        :param normal_line: 直线一般式
        :param left_pts:    左侧点集
        :param right_pts:   右侧点集
        :return: 点A和点B的欧式距离
        """
        A,B,C = normal_line
        left_pt = None
        right_pt = None
        left_diff_min = float('+Inf')
        right_diff_min = float('+Inf')
        for pt in left_pts:
            x, y = pt
            diff = abs(A*x+B*y+C)
            if diff < 1 and diff < left_diff_min:
                # pt on line
                left_pt = pt
                left_diff_min = diff
        for pt in right_pts:
            x, y = pt
            diff = abs(A*x+B*y+C)
            if diff < 1 and diff < right_diff_min:
                # pt on line
                right_pt = pt
                right_diff_min = diff
        if left_pt is not None and right_pt is not None:
            return euc_dis(left_pt, right_pt), left_pt, right_pt
        else:
            return None, None, None

    @staticmethod
    def _fit_edge(pts, y_max, y_min=0):
        """
        对点集拟合直线段

        待拟合的直线近似垂直于x轴，因此:
        x=y*k+b
        :param pts: 点集合
        :param y_max: 直线段上端点坐标y
        :param y_min: 直线段下端点坐标y
        :return: 直线段Edge
        """

        def line_func(y, k, b):
            return y * k + b
        # fit left edge
        pt_xs = [pt[0] for pt in pts]
        pt_ys = [pt[1] for pt in pts]

        # x_diffs = [abs(x-pt_xs[0]) for x in pt_xs]
        # if sum(x_diffs) == 0:
        #     # pts的x坐标完全相同，避免curve_fit警告
        #     pts = sorted(pts, key=lambda pt: pt[1])
        #     return Edge(pts[0], pts[-1])

        k, b = curve_fit(line_func, pt_ys, pt_xs)[0]
        # print('x = %.2fy + %.2f' % (k, b))
        pt_bot = [y_min*k+b, y_min]
        pt_top = [y_max*k+b, y_max]
        line = Edge(pt_top, pt_bot)
        return line

    def _patch_width(self, sub_mask):
        h, w = sub_mask.shape

        l_pts, r_pts = self._get_edge_pts(sub_mask)
        if len(l_pts) < 2 or len(r_pts) < 2:
            return None, None

        l_pts = sorted(l_pts, key=lambda pt: (pt[1], pt[0]))
        r_pts = sorted(r_pts, key=lambda pt: (pt[1], pt[0]))

        l_line = self._fit_edge(l_pts, h)
        r_line = self._fit_edge(r_pts, h)

        l_error = line_estimate_error(l_line.normal(), l_pts)
        r_error = line_estimate_error(r_line.normal(), r_pts)
        # print('L(%.2f) | R(%.2f)' % (l_error, r_error))
        if l_error is None or r_error is None or l_error > 3 or r_error > 3:
            return None, None

        corners = {
            'left_top': l_pts[0],
            'right_top': r_pts[0],
            'left_bottom': l_pts[-1],
            'right_bottom': r_pts[-1],
        }

        # debug
        im_sub_mask = np.stack((sub_mask, sub_mask, sub_mask), axis=2)
        im_sub_mask[im_sub_mask > 0] = 255
        im_sub_mask = im_sub_mask.astype(np.uint8)
        x1 = int(l_line.get_pt(0)[0])
        y1 = int(l_line.get_pt(0)[1])
        x2 = int(l_line.get_pt(1)[0])
        y2 = int(l_line.get_pt(1)[1])
        cv2.line(im_sub_mask, (x1, y1), (x2, y2), (0, 0, 255), 1)
        x1 = int(r_line.get_pt(0)[0])
        y1 = int(r_line.get_pt(0)[1])
        x2 = int(r_line.get_pt(1)[0])
        y2 = int(r_line.get_pt(1)[1])
        cv2.line(im_sub_mask, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # visualize_image(im_sub_mask, 'sub1_%d' % np.random.randint(0, 1000), '2193')

        alpha = angle(l_line.vec(), r_line.vec())
        if alpha < 5:
            # 两条直线接近平行，检测正常
            # 计算距离
            pts = l_line.get_pts(10)
            l2r_dis = []
            for pt in pts:
                l_norm_per = l_line.normal_perpendicular(pt)
                if l_norm_per is None:
                    continue
                wid, _, _ = self._width(l_norm_per, l_pts, r_pts)
                if wid is not None:
                    l2r_dis.append(wid)

            pts = r_line.get_pts(10)
            r2l_dis = []
            for pt in pts:
                r_norm_per = r_line.normal_perpendicular(pt)
                if r_norm_per is None:
                    continue
                wid, lpt, rpt = self._width(r_norm_per, l_pts, r_pts)
                if wid is not None:
                    r2l_dis.append(wid)
                    cv2.line(im_sub_mask, tuple(lpt), tuple(rpt), (0, 255, 0), 1)
            # visualize_image(im_sub_mask, 'sub2_%d' % np.random.randint(0, 1000), '2193')
            # 计算均值
            dis_arr = l2r_dis + r2l_dis
            if len(dis_arr) > 0:
                return sum(dis_arr)*1.0/len(dis_arr), corners
            else:
                return None, None
        else:
            return None, None

    def is_seg_succ(self):
        h, w = self.contour_mask.shape
        ys, xs = np.where(self.contour_mask > 0)
        xmin = min(xs)
        xmax = max(xs)
        if xmin < 20 and (w - xmax) < 20:
            # 树太粗
            return False

        # 获取两条边缘上的点
        l_pts, r_pts = self._get_edge_pts(self.contour_mask)
        if len(l_pts) < 2 or len(r_pts) < 2:
            return False

        # 分别拟合直线
        l_line = self._fit_edge(l_pts, h)
        r_line = self._fit_edge(r_pts, h)

        # debug
        # im_sub_mask = np.stack((self.contour_mask, self.contour_mask, self.contour_mask), axis=2)
        # im_sub_mask = im_sub_mask.astype(np.uint8)
        # x1 = int(l_line.get_pt(0)[0])
        # y1 = int(l_line.get_pt(0)[1])
        # x2 = int(l_line.get_pt(1)[0])
        # y2 = int(l_line.get_pt(1)[1])
        # cv2.line(im_sub_mask, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # x1 = int(r_line.get_pt(0)[0])
        # y1 = int(r_line.get_pt(0)[1])
        # x2 = int(r_line.get_pt(1)[0])
        # y2 = int(r_line.get_pt(1)[1])
        # cv2.line(im_sub_mask, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # show_image(im_sub_mask, 'l_r_line')

        alpha = angle(l_line.vec(), r_line.vec())
        if alpha < 20:
            # 两条直线接近平行，检测正常
            return True
        else:
            return False

    def pixel_width(self):
        # 计算树干直径
        h, w = self.contour_mask.shape

        n_patch = 4
        patch_widths = []
        patch_left_tops = []
        patch_left_bottoms = []
        patch_right_tops = []
        patch_right_bottoms = []
        interval_y = h * 1.0 / n_patch
        # 将轮廓mask分成4份，分别计算直径
        # 从上往下，分四块
        # 000000
        # 111111
        # 222222
        # 333333
        for i in range(n_patch):
            # 对每一块
            sub_mask = self.contour_mask[int(i*interval_y):int((i+1)*interval_y), :]
            # visualize_image(sub_mask, 'sub_%d' % i, '2193')
            patch_width, sub_patch_corners = self._patch_width(sub_mask)
            if patch_width is not None:
                patch_widths.append(patch_width)

                # fix shift on y
                patch_left_top = [sub_patch_corners['left_top'][0],
                                  sub_patch_corners['left_top'][1] + int(i*interval_y)]
                patch_right_top = [sub_patch_corners['right_top'][0],
                                   sub_patch_corners['right_top'][1] + int(i*interval_y)]
                patch_left_bottom = [sub_patch_corners['left_bottom'][0],
                                     sub_patch_corners['left_bottom'][1] + int(i*interval_y)]
                patch_right_bottom = [sub_patch_corners['right_bottom'][0],
                                      sub_patch_corners['right_bottom'][1] + int(i*interval_y)]

                patch_left_tops.append(patch_left_top)
                patch_right_tops.append(patch_right_top)
                patch_left_bottoms.append(patch_left_bottom)
                patch_right_bottoms.append(patch_right_bottom)
        # 计算均值
        if len(patch_widths) > 0:
            # pixel width, score
            trunk_corners = {
                'left_top': patch_left_tops[0],
                'right_top': patch_right_tops[0],
                'left_bottom': patch_left_bottoms[-1],
                'right_bottom': patch_right_bottoms[-1]
            }
            conf = 1 - 0.1 * (n_patch - len(patch_widths))
            width = sum(patch_widths)*1.0/len(patch_widths)
            return width, conf, trunk_corners
        else:
            return None, None, None

    def real_width_v1(self, shot_distance, RP_ratio):
        # W = M * (X/(X-M))
        # W为实际树径
        # M为带误差树径
        # X为Tag所在平面与焦点的距离
        pixel_width = self.pixel_width()
        if pixel_width is not None and shot_distance is not None:
            M = (pixel_width * RP_ratio)
            X = shot_distance * 1.0
            width = M * (X / (X - M))
            return width
        else:
            return None

    def real_width_v2(self, shot_distance, RP_ratio):
        pixel_width, conf, trunk_corners = self.pixel_width()
        if pixel_width is not None and RP_ratio is not None:
            M = (pixel_width * RP_ratio)
            beta = math.atan(M / shot_distance)
            sin_beta = math.sin(beta)
            width = shot_distance * sin_beta / (1 - sin_beta)
            return M, conf, trunk_corners
        else:
            return -1, -1, None