# coding: utf-8
# status: reviewed

import math


def eight_connected(pt1, pt2):
    """
    判断两个坐标是否8连通
    :param pt1:
    :param pt2:
    :return:
    """
    if abs(pt1[0] - pt2[0]) <= 1 and abs(pt1[1] - pt2[1]) <= 1:
        return True
    else:
        return False


def pt_2_line(normal_line, pt):
    """
    点到直线距离
    :param normal_line: 直线一般式
    :param pt: 点坐标
    :return: 距离
    """
    if normal_line is not None and pt is not None:
        x, y = pt
        A, B, C = normal_line
        dis = abs(A*x + B*y + C) * 1.0 / ((A**2 + B**2) ** 0.5)
        return dis
    else:
        return None


def signed_pt_2_line(normal_line, pt):
    """
    点到直线距离（带符号）
    :param normal_line: 直线一般式
    :param pt: 点坐标
    :return: 带符号距离
    """
    if normal_line is not None and pt is not None:
        x, y = pt
        A, B, C = normal_line
        dis = (A*x + B*y + C) * 1.0 / ((A ** 2 + B ** 2) ** 0.5)
        return dis
    else:
        return None


def line_estimate_error(normal_line, pts):
    """
    回归直线的平均距离误差
    :param normal_line: 回归得到的直线一般式
    :param pts: 用于回归直线的点集
    :return: 平均距离误差
    """
    if normal_line is not None and pts is not None and len(pts) > 0:
        error_sum = 0
        for pt in pts:
            error_sum += pt_2_line(normal_line, pt)
        return error_sum * 1.0 / len(pts)
    else:
        return None


def cos(v1, v2):
    """
    余弦定理
    计算两向量夹角cosine
    :param v1: 向量1[x, y]
    :param v2: 向量2[x, y]
    :return: cos值
    """
    cos_alpha = (v1[0] * v2[0] + v1[1] * v2[1]) / ((v1[0] ** 2 + v1[1] ** 2) ** 0.5 * (v2[0] ** 2 + v2[1] ** 2) ** 0.5)
    cos_alpha = min(cos_alpha, 1.0)
    cos_alpha = max(cos_alpha, -1.0)
    return cos_alpha


def angle(v1, v2):
    """
    计算两向量夹角[0', 90'](角度制)
    :param v1:
    :param v2:
    :return: 夹角
    """

    # 计算两条直线的夹角的cosine
    c = abs(cos(v1, v2))
    # 计算夹角（角度制）
    alpha = math.acos(c) * (180 / math.pi)
    alpha = max(0, alpha)
    alpha = min(90, alpha)
    return alpha


def euc_dis(pt1, pt2):
    """
    计算两点欧式距离
    :param pt1: 点1坐标
    :param pt2: 点2坐标
    :return:
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return ((x1-x2)**2+(y1-y2)**2)**0.5


def cal_cross_pt(l1_n, l2_n):
    """
    计算两条直线的交点
    参数为直线一般式：Ax+By+C=0 [A,B,C]
    :param l1_n: l1一般式
    :param l2_n: l2一般式
    :return:
    """

    def cal_cross_pt_kb(k1, b1, k2, b2):
        """
        计算两条直线（斜截式）的交点
        :param k1: l1斜率
        :param b1: l1截距
        :param k2: l2斜率
        :param b2: l2截距
        :return: 焦点坐标
        """
        k1 = float(k1)
        b1 = float(b1)
        k2 = float(k2)
        b2 = float(b2)

        if k1 == k2:
            # 平行，不存在交点
            return None
        else:
            cross_x = (b2 - b1) / (k1 - k2)
            cross_y = k1 * cross_x + b1
            return [cross_x, cross_y]

    # [A,B,C]
    # Ax+By+C=0
    if l1_n is None or l2_n is None or len(l1_n) == len(l2_n) == 3:
        # 输入不合法
        return None

    l1_n = [float(v) for v in l1_n]
    l2_n = [float(v) for v in l2_n]

    A1,B1,C1 = l1_n
    A2,B2,C2 = l2_n

    if A1 == B1 == 0 or A2 == B2 == 0:
        # l1 或 l2 不是直线
        return None

    if B1 == 0 and B2 == 0:
        # 平行且垂直于x轴，无交点
        return None
    elif B1 == 0 and B2 != 0:
        # l1垂直于x轴，l2不垂直于x轴
        cross_x = (-C1)/A1
        cross_y = (-C2-A2*cross_x)/B2
        return [cross_x, cross_y]
    elif B1 != 0 and B2 == 0:
        # l2垂直于x轴，l1不垂直于x轴
        cross_x = (-C2)/A2
        cross_y = (-C1-A1*cross_x)/B1
        return [cross_x, cross_y]
    else:
        # l1 l2都不垂直于x轴
        # l1 l2都可以表示为斜截式
        k1 = (-A1)/B1
        b1 = (-C1)/B1
        k2 = (-A2)/B2
        b2 = (-C2)/B2
        return cal_cross_pt_kb(k1,b1,k2,b2)


class Edge:

    def __init__(self, pt1, pt2):
        """
        初始化Edge
        :param pt1: 端点1[x,y]
        :param pt2: 端点2[x,y]
        """
        # pt1和pt2不能是同一个点
        assert ~((pt1[0]-pt2[0]) == 0 and (pt1[1]-pt2[1]) == 0)
        self.pts = [pt1, pt2]
        self.merged = False
        self.connects = [None, None]

    def get_connect(self, ind):
        return self.connects[ind]

    def set_connect(self, pt_ind, edge, e_pt_ind):
        self.connects[pt_ind] = (edge, e_pt_ind)

    def vec(self):
        """
        Edge的方向向量
        :return: 方向向量
        """
        x1, y1 = self.pts[0]
        x2, y2 = self.pts[1]
        return [x2-x1, y2-y1]

    def get_pt(self, ind):
        return self.pts[ind]

    def length(self):
        """
        Edge长度
        :return: 两端点欧式距离
        """
        return euc_dis(self.pts[0], self.pts[1])

    def k(self):
        """
        Edge所在直线的斜率
        :return: 斜率
        """
        x1, y1 = self.pts[0]
        x2, y2 = self.pts[1]
        if (x2-x1) != 0:
            # (y2-y1)/(x2-x1)
            return 1.0*(y2-y1)/(x2-x1)
        else:
            # x1==x2,Edge垂直于x轴，k不存在
            return None

    def b(self):
        """
        Edge所在直线的截距
        :return: 截距
        """
        x, y = self.pts[0]
        if self.k() is not None:
            # b=y-kx
            return y-self.k()*x
        else:
            # x1==x2,Edge垂直于x轴，b不存在
            return None

    def normal(self):
        """
        Edge所在直线的一般式
        Ax+By+C=0
        :return: [A,B,C]
        """
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
        return [A, B, C]

    def normal_perpendicular(self, pt):
        """
        Edge所在直线上一点pt，且垂直于Edge的直线一般式
        :param pt: 任意一点
        :return: 垂线一般式
        """
        A,B,C = self.normal()
        x0, y0 = pt
        if abs(A*x0+B*y0+C) < 1e-5:
            # pt在自身所在直线上
            if A == 0:
                # 自身垂直于y轴
                # x = x0
                Ap = 1
                Bp = 0
                Cp = -x0
            elif B == 0:
                # 自身垂直于x轴
                # y = y0
                Ap = 0
                Bp = 1
                Cp = -y0
            else:
                # 用斜截式表示
                k = self.k()
                kp = -1.0/k
                bp = y0-kp*x0
                Ap = kp
                Bp = -1
                Cp = bp
            return [Ap, Bp, Cp]
        else:
            # pt不在Edge所在直线上
            return None

    def get_pts(self, pt_num):
        """
        获得Edge上的pt_num个点坐标
        :param pt_num: 点个数
        :return: 坐标列表
        """
        if pt_num <= 0:
            return None
        else:
            x1, y1 = self.get_pt(0)
            x2, y2 = self.get_pt(1)
            pts = []
            A,B,C = self.normal()
            if abs(x1-x2) >= abs(y1-y2):
                # 沿着x轴方向采点
                assert B != 0
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                interval = (xmax-xmin)*1.0/pt_num
                for i in range(pt_num):
                    x = xmin + interval*i
                    y = (-C-A*x)*1.0/B
                    pts.append([x, y])
            else:
                # 沿着y轴方向采点
                assert A != 0
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                interval = (ymax-ymin)*1.0/pt_num
                for i in range(pt_num):
                    y = ymin + interval * i
                    x = (-C-B*y)*1.0/A
                    pts.append([x, y])
            return pts