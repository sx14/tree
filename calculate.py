# coding: utf-8
import os
import traceback
import argparse

from measure.common.geo_utils import angle, parallel_distance
from measure.trunk.cal_trunk import Edge
from measure.calibrator.calibrator_factory import get_calibrator
from util.resize_image import *
from util.result import Result, InfoEnum
from util.show_image import *
from util.my_io import *
from util.my_logger import save_log

DEBUG = False
SHOW = False


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure')
    parser.add_argument('-i', '--image_path',   help='Please provide image path.',          required=True)
    parser.add_argument('-p', '--points',       help='Please provide annotation points.',   required=True)
    parser.add_argument('-o', '--output',       help='Please provide output file path.')
    args = parser.parse_args()
    return args


def measure_tree_width(image_path, trunk_corners):
    result = Result()  # 保存当前图片的分割结果，初始化为失败
    result.set_image_path(image_path)
    result.set_time(0)

    if not os.path.exists(image_path):
        result.set_info(InfoEnum.IMAGE_NOT_EXIST)
        return result
    if trunk_corners is None:
        result.set_info(InfoEnum.BAD_MANUAL_ANNO)
        return result

    result.set_trunk_left_top(trunk_corners['left_top'])
    result.set_trunk_left_bottom(trunk_corners['left_bottom'])
    result.set_trunk_right_top(trunk_corners['right_top'])
    result.set_trunk_left_bottom(trunk_corners['right_bottom'])

    im = cv2.imread(image_path)  # 加载图片
    im, resize_ratio = resize_image_with_ratio(im)  # 调整尺寸

    # 检测标定物
    calibrator = get_calibrator(im, 0, DEBUG and SHOW)
    if calibrator is None:
        result.set_info(InfoEnum.CALIBRATOR_DET_FAILED)
        return result

    # 在结果中保存激光点坐标
    # resize坐标 -> 原始坐标
    org_calibrate_pts = []
    for calibrate_pt in calibrator.get_calibrate_points():
        org_calibrate_pt = recover_coordinate(calibrate_pt, resize_ratio)
        org_calibrate_pts.append(org_calibrate_pt)
    result.set_calibrate_points(org_calibrate_pts)

    # 计算树径
    trunk_left_top = resize_coordinate(trunk_corners['left_top'], resize_ratio)
    trunk_left_bottom = resize_coordinate(trunk_corners['left_bottom'], resize_ratio)
    trunk_right_top = resize_coordinate(trunk_corners['right_top'], resize_ratio)
    trunk_right_bottom = resize_coordinate(trunk_corners['right_bottom'], resize_ratio)

    l_line = Edge(trunk_left_top, trunk_left_bottom)
    r_line = Edge(trunk_right_top, trunk_right_bottom)

    alpha = angle(l_line.vec(), r_line.vec())
    if alpha < 5:
        pixel_dis_l2r = parallel_distance(l_line, r_line)
        pixel_dis_r2l = parallel_distance(r_line, l_line)
        if pixel_dis_r2l is not None and pixel_dis_r2l is not None:
            pixel_width = (pixel_dis_l2r + pixel_dis_r2l) / 2.0
            RP_ratio = calibrator.RP_ratio()

            real_width = (pixel_width * RP_ratio)
            result.set_info(InfoEnum.SUCCESS)
            result.set_width(real_width)
            result.set_conf(1.0)
        else:
            result.set_info(InfoEnum.BAD_MANUAL_ANNO)
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
    try:
        args = parse_args()
        image_path = args.image_path
        trunk_corners = parse_points(args.points)
        result = measure_tree_width(image_path, trunk_corners)
        print_json(result.get_result())

        output_path = args.output
        if output_path is not None:
            save_path = args.output
            save_results(result, save_path)
    except:
        if 'args' in dir():
            save_log(args.__str__() + '\n' + traceback.format_exc())
        else:
            save_log(traceback.format_exc())
