# coding: utf-8
import os
from det.common.geo_utils import euc_dis, angle
from det.trunk.cal_trunk import Edge
from det.calibrator.laser import get_laser_points
from det.calibrator.laser import Laser
from util.resize_image import *
from util.result import Result, InfoEnum
from util.show_image import *
from util.my_io import *
import argparse

DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure V1.1')
    parser.add_argument('-i', '--image_path',   help='Please provide image path.',          required=True)
    parser.add_argument('-p', '--points',       help='Please provide annotation points.',   required=True)
    parser.add_argument('-o', '--output',       help='Please provide output file path.',    default='None')
    args = parser.parse_args()
    return args


def measure_tree_width(image_path, trunk_corners):
    result = Result()  # 保存当前图片的分割结果，初始化为失败

    if not os.path.exists(image_path):
        result.set_info(InfoEnum.IMAGE_NOT_EXIST)
        return result

    im = cv2.imread(image_path)  # 加载图片
    im, resize_ratio = resize_image_with_ratio(im)  # 调整尺寸

    # step1: 获得激光点对位置，激光点mask，激光mask，激光得分
    pt_pair, pt_mask, laser_mask, pt_conf = get_laser_points(im, DEBUG)
    if DEBUG:
        show_images([im, laser_mask, pt_mask])
        pass

    if len(pt_pair) != 2:
        result.set_info(InfoEnum.LASER_DET_FAILED)
        if DEBUG:
            print(InfoEnum.LASER_DET_FAILED)
        return result
    else:
        if DEBUG:
            print('Laser point pair detection success.')
            pass

    laser = Laser(pt_pair, pt_mask, laser_mask)

    # 在结果中保存激光点坐标
    # resize坐标 -> 原始坐标
    laser_top_pt = recover_coordinate(laser.get_top_pt(), resize_ratio)
    laser_bottom_pt = recover_coordinate(laser.get_bottom_pt(), resize_ratio)
    result.set_laser_top(laser_top_pt)
    result.set_laser_bottom(laser_bottom_pt)

    # 计算树径
    trunk_left_top = resize_coordinate(trunk_corners['left_top'], resize_ratio)
    trunk_left_bottom = resize_coordinate(trunk_corners['left_bottom'], resize_ratio)
    trunk_left_mid = [(trunk_left_top[0] + trunk_left_bottom[0]) / 2.0,
                      (trunk_left_top[1] + trunk_left_bottom[1]) / 2.0]
    trunk_right_top = resize_coordinate(trunk_corners['right_top'], resize_ratio)
    trunk_right_bottom = resize_coordinate(trunk_corners['right_bottom'], resize_ratio)
    trunk_right_mid = [(trunk_right_top[0] + trunk_right_bottom[0]) / 2.0,
                      (trunk_right_top[1] + trunk_right_bottom[1]) / 2.0]

    l_line = Edge(trunk_left_top, trunk_left_bottom)
    r_line = Edge(trunk_right_top, trunk_right_bottom)

    alpha = angle(l_line.vec(), r_line.vec())
    if alpha < 10:
        pixel_dis_top = euc_dis(trunk_left_top, trunk_right_top)
        pixel_dis_bot = euc_dis(trunk_left_bottom, trunk_right_bottom)
        pixel_dis_mid = euc_dis(trunk_left_mid, trunk_right_mid)

        pixel_width = (pixel_dis_top + pixel_dis_mid + pixel_dis_bot) / 3.0
        RP_ratio = laser.RP_ratio()

        # TODO: 此处计算为粗略近似值
        real_width = (pixel_width * RP_ratio)
        result.set_info(InfoEnum.SUCCESS)
        result.set_trunk_left_top(trunk_corners['left_top'])
        result.set_trunk_left_bottom(trunk_corners['left_bottom'])
        result.set_trunk_right_top(trunk_corners['right_top'])
        result.set_trunk_right_bottom(trunk_corners['right_bottom'])
        result.set_width(real_width)
        result.set_conf(1.0)
    else:
        result.set_info(InfoEnum.BAD_MANUAL_ANNO)

    return result


def parse_points(raw_points):
    # x1,y1,x2,y2,x3,y3,x4,y4
    raw_values = raw_points.split(',')
    values = [int(raw_v) for raw_v in raw_values]
    trunk_corners = None
    if len(raw_values) == 8:
        points = []
        for i in range(4):
            points.append([values[i*2], values[i*2+1]])
            # 先按y排序
            points = sorted(points, key=lambda pt: pt[1])
            # 再按x排序
            top_pts = sorted(points[:2], key=lambda pt: pt[0])
            bottom_pts = sorted(points[2:], key=lambda pt: pt[0])
            trunk_corners = {
                'left_top': top_pts[0],
                'left_bottom': bottom_pts[0],
                'right_top': top_pts[1],
                'right_bottom': bottom_pts[1]}
    return trunk_corners


def load_input(input_path):
    image_path = None
    trunk_corners = None
    if os.path.exists(input_path):
        import json
        with open(input_path) as f:
            input_data = json.load(f)
            if 'image_path' in input_data:
                image_path = input_data['image_path']
            if 'points' in input_data and len(input_data['points']) == 4:
                points = input_data['points']
                # 先按y排序
                points = sorted(points, key=lambda pt: pt[1])
                # 再按x排序
                top_pts = sorted(points[:2], key=lambda pt: pt[0])
                bottom_pts = sorted(points[2:], key=lambda pt: pt[0])
                trunk_corners = {
                    'left_top': top_pts[0],
                    'left_bottom': bottom_pts[0],
                    'right_top': top_pts[1],
                    'right_bottom': bottom_pts[1]}
    return image_path, trunk_corners


if __name__ == '__main__':
    args = parse_args()
    image_path = args.image_path
    trunk_corners = parse_points(args.points)
    result = measure_tree_width(image_path, trunk_corners)
    print_json(result)

    output_path = args.output
    if output_path != 'None':
        save_path = args.output
        save_results(result, save_path)
