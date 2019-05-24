# coding: utf-8
import math


def eight_connected(pt1, pt2):
    if abs(pt1[0] - pt2[0]) <= 1 and abs(pt1[1] - pt2[1]) <= 1:
        return True
    else:
        return False

def pt_2_line(normal_line, pt):
    if normal_line is not None and pt is not None:
        x, y = pt
        A, B, C = normal_line
        dis = abs(A*x + B*y + C) * 1.0 / ((A**2 + B**2) ** 0.5)
        return dis
    else:
        return None


def signed_pt_2_line(normal_line, pt):
    if normal_line is not None and pt is not None:
        x, y = pt
        A, B, C = normal_line
        dis = (A*x + B*y + C) * 1.0 / ((A ** 2 + B ** 2) ** 0.5)
        return dis
    else:
        return None


def line_estimate_error(normal_line, pts):
    if normal_line is not None and pts is not None and len(pts) > 0:
        error_sum = 0
        for pt in pts:
            error_sum += pt_2_line(normal_line, pt)
        return error_sum * 1.0 / len(pts)
    else:
        return float('+Inf')


def cos(v1, v2):
    # 计算两向量夹角cosine
    cos_alpha = (v1[0] * v2[0] + v1[1] * v2[1]) / ((v1[0] ** 2 + v1[1] ** 2) ** 0.5 * (v2[0] ** 2 + v2[1] ** 2) ** 0.5)
    cos_alpha = min(cos_alpha, 1.0)
    cos_alpha = max(cos_alpha, -1.0)
    return cos_alpha


def angle(v1, v2):
    # 计算两向量夹角[0', 90'](角度制)

    # 计算两条直线的夹角的cosine
    c = abs(cos(v1, v2))
    # 计算夹角（角度制）
    alpha = math.acos(c) * (180 / math.pi)
    alpha = max(0, alpha)
    alpha = min(90, alpha)
    return alpha


def euc_dis(pt1, pt2):
    # pt: [x,y]
    x1, y1 = pt1
    x2, y2 = pt2
    return ((x1-x2)**2+(y1-y2)**2)**0.5


def cal_cross_pt(l1_n, l2_n):
    # 计算两条直线的交点
    # 参数为直线一般式：Ax+By+C=0 [A,B,C]

    def cal_cross_pt_kb(k1, b1, k2, b2):
        k1 = float(k1)
        b1 = float(b1)
        k2 = float(k2)
        b2 = float(b2)

        if k1 == k2:
            return None
        else:
            cross_x = (b2 - b1) / (k1 - k2)
            cross_y = k1 * cross_x + b1
            return [cross_x, cross_y]

    # [A,B,C]
    # Ax+By+C=0
    assert len(l1_n) == len(l2_n) == 3
    for i in range(len(l1_n)):
        l1_n[i] = float(l1_n[i])
        l2_n[i] = float(l2_n[i])

    A1,B1,C1 = l1_n
    A2,B2,C2 = l2_n
    if B1 == 0 and B2 == 0:
        # 平行
        return None
    elif B1 == 0 and B2 != 0:
        cross_x = (-C1)/A1
        cross_y = (-C2-A2*cross_x)/B2
        return [cross_x, cross_y]
    elif B1 != 0 and B2 == 0:
        cross_x = (-C2)/A2
        cross_y = (-C1-A1*cross_x)/B1
        return [cross_x, cross_y]
    else:
        k1 = (-A1)/B1
        b1 = (-C1)/B1
        k2 = (-A2)/B2
        b2 = (-C2)/B2
        return cal_cross_pt_kb(k1,b1,k2,b2)


class Edge:
    def __init__(self, pt1, pt2):
        assert ~((pt1[0]-pt2[0]) == 0 and (pt1[1]-pt2[1]) == 0)
        self.pts = [pt1, pt2]
        self.connects = [None, None]
        self.merged = False

    def vec(self):
        x1, y1 = self.pts[0]
        x2, y2 = self.pts[1]
        return [x2-x1, y2-y1]
        # return self.pts[1] - self.pts[0]

    def get_pt(self, ind):
        return self.pts[ind]

    def get_connect(self, ind):
        return self.connects[ind]

    def set_connect(self, my_pt_ind, edge, e_pt_ind):
        self.connects[my_pt_ind] = (edge, e_pt_ind)

    def length(self):
        return euc_dis(self.pts[0], self.pts[1])

    def k(self):
        x1, y1 = self.pts[0]
        x2, y2 = self.pts[1]
        if (x2-x1) != 0:
            # (y2-y1)/(x2-x1)
            return 1.0*(y2-y1)/(x2-x1)
        else:
            return None

    def b(self):
        x, y = self.pts[0]
        if self.k() is not None:
            # b=y-kx
            return y-self.k()*x
        else:
            return None

    def normal(self):
        # Ax+By+C=0
        x1, y1 = self.pts[0]
        x2, y2 = self.pts[1]
        if x1 == x2:
            A = 1
            B = 0
            C = -x1

        elif y1 == y2:
            A = 0
            B = 1
            C = -y1

        else:
            A = self.k()
            B = -1
            C = self.b()
        return [A,B,C]

    def normal_perpendicular(self, pt):
        # 过pt的垂线一般式
        A,B,C = self.normal()
        x,y = pt
        if abs(A*x+B*y+C) < 1e-5:
            if A == 0:
                Ap = 1
                Bp = 0
                Cp = -x
            elif B == 0:
                Ap = 0
                Bp = 1
                Cp = -y
            else:
                k = self.k()
                kp = -1.0/k
                bp = y-kp*x
                Ap = kp
                Bp = -1
                Cp = bp
            return [Ap, Bp, Cp]
        else:
            # pt not on line
            return None

    def get_pts(self, pt_num):
        if pt_num <= 0:
            return None, None
        else:
            x1, y1 = self.get_pt(0)
            x2, y2 = self.get_pt(1)
            pts = []
            A,B,C = self.normal()
            if abs(x1-x2) >= abs(y1-y2):
                assert B != 0
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                interval = (xmax-xmin)*1.0/pt_num
                for i in range(pt_num):
                    x = xmin + interval*i
                    y = (-C-A*x)*1.0/B
                    pts.append([x, y])
            else:
                assert A != 0
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                interval = (ymax-ymin)*1.0/pt_num
                for i in range(pt_num):
                    y = ymin + interval * i
                    x = (-C-B*y)*1.0/A
                    pts.append([x, y])
            return pts