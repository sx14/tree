# coding: utf-8
from scipy.optimize import curve_fit
from det.common.det_edge import *
from det.common.geo_utils import *


class Trunk:
    def __init__(self, trunk_mask):
        self.trunk_mask = trunk_mask
        self.contour_mask = detect_contour(trunk_mask)
        h, w = self.contour_mask.shape
        self.contour_mask = self.contour_mask[4:h - 4, :]

    def _get_edge_pts(self, sub_mask):
        ys, xs = np.where(sub_mask > 0)
        return [[xs[i], ys[i]] for i in range(len(xs))]

    def _get_edge_pts1(self, sub_mask):
        ys, xs = np.where(sub_mask > 0)
        pts = [[xs[i], ys[i]] for i in range(len(xs))]
        pts = sorted(pts, key=lambda a: (a[1], a[0]))
        if len(pts) == 0:
            return None, None

        l_pts = []
        l_pt_ind = 0
        while l_pt_ind >= 0:
            l_pt = pts[l_pt_ind]
            l_pts.append(l_pt)
            pts.pop(l_pt_ind)
            l_pt_ind = -1
            for i, pt in enumerate(pts):
                if eight_connected(pt, l_pt):
                    l_pt_ind = i
                    break
        r_pts = pts
        return l_pts, r_pts

    def _get_edge_pts2(self, sub_mask):
        ys, xs = np.where(sub_mask > 0)
        pts = [[xs[i], ys[i]] for i in range(len(xs))]
        if len(pts) == 0:
            return [], []
        # sorted by: y-first; x-second
        pts = sorted(pts, key=lambda a: (a[1], a[0]))
        l_pt_q = [pts.pop(0)]   # 左边缘上顶点
        l_pts = []
        while len(l_pt_q) > 0:
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
            for l_pt_ind in l_pt_inds:
                pts.pop(l_pt_ind)
        r_pts = pts
        return l_pts, r_pts

    def _width(self, normal_line, left_pts, right_pts):
        A,B,C = normal_line
        left_pt = None
        right_pt = None
        left_diff = float('+Inf')
        right_diff = float('Inf')
        for pt in left_pts:
            x, y = pt
            diff = abs(A*x+B*y+C)
            if diff < 1 and diff < left_diff:
                # pt on line
                left_pt = pt
                left_diff = diff
        for pt in right_pts:
            x, y = pt
            diff = abs(A*x+B*y+C)
            if diff < 1 and diff < right_diff:
                # pt on line
                right_pt = pt
                right_diff = diff
        if left_pt is not None and right_pt is not None:
            return euc_dis(left_pt, right_pt)
        else:
            return None

    def _fit_edge(self, pts, y_max):
        # 拟合的直线近似垂直于x轴，因此:
        # x=y*k+b
        def line_func(y, k, b):
            return y * k + b
        # fit left edge
        pt_xs = [pt[0] for pt in pts]
        pt_ys = [pt[1] for pt in pts]
        k, b = curve_fit(line_func, pt_ys, pt_xs)[0]
        pt_bot = [0*k+b, 0]
        pt_top = [y_max*k+b, y_max]
        line = Edge(pt_top, pt_bot)
        return line

    def _patch_width(self, sub_mask):
        h, w = sub_mask.shape

        l_pts, r_pts = self._get_edge_pts2(sub_mask)
        if len(l_pts) < 2 or len(r_pts) < 2:
            return None

        l_line = self._fit_edge(l_pts, h)
        r_line = self._fit_edge(r_pts, h)

        l_error = line_estimate_error(l_line.normal(), l_pts)
        r_error = line_estimate_error(r_line.normal(), r_pts)
        # print('L(%.2f) | R(%.2f)' % (l_error, r_error))
        if l_error > 3 or r_error > 3:
            return None

        # debug
        im_sub_mask = np.stack((sub_mask, sub_mask, sub_mask), axis=2)
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
        # show_image(im_sub_mask)
        # cv2.imwrite('%d.jpg' % (np.random.randint(0, 1000)), im_sub_mask)

        alpha = angle(l_line.vec(), r_line.vec())
        if alpha < 5:
            # 两条直线接近平行，检测正常
            # 计算距离
            pts = l_line.get_pts(10)
            l2r_dis = []
            for pt in pts:
                Alp, Blp, Clp = l_line.normal_perpendicular(pt)
                wid = self._width([Alp, Blp, Clp], l_pts, r_pts)
                if wid is not None:
                    l2r_dis.append(wid)

            pts = r_line.get_pts(10)
            r2l_dis = []
            for pt in pts:
                Arp, Brp, Crp = r_line.normal_perpendicular(pt)
                wid = self._width([Arp, Brp, Crp], l_pts, r_pts)
                if wid is not None:
                    r2l_dis.append(wid)
            # 计算均值
            dis_arr = l2r_dis + r2l_dis
            if len(dis_arr) > 0:
                return sum(dis_arr)*1.0/len(dis_arr)
            else:
                return None
        else:
            return None

    def is_seg_succ(self):
        h, w = self.contour_mask.shape
        ys, xs = np.where(self.contour_mask > 0)
        xmin = min(xs)
        xmax = max(xs)
        if xmin < 20 and (w - xmax) < 20:
            # 树太粗
            return False

        # 获取两条边缘上的点
        l_pts, r_pts = self._get_edge_pts2(self.contour_mask)
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
        interval_y = h * 1.0 / n_patch
        # 将轮廓mask分成4份，分别计算直径
        for i in range(n_patch):
            sub_mask = self.contour_mask[int(i*interval_y):int((i+1)*interval_y), :]
            patch_width = self._patch_width(sub_mask)
            if patch_width is not None:
                patch_widths.append(patch_width)
        # 计算均值
        if len(patch_widths) > 0:
            # pixel width, score
            return sum(patch_widths)*1.0/len(patch_widths), (1 - 0.1 * (n_patch - len(patch_widths)))
        else:
            return -1, -1

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
        pixel_width, conf = self.pixel_width()
        if pixel_width is not None and RP_ratio is not None:
            M = (pixel_width * RP_ratio)
            beta = math.atan(M / shot_distance)
            sin_beta = math.sin(beta)
            width = shot_distance * sin_beta / (1 - sin_beta)
            return M, conf
        else:
            return -1, -1