# coding: utf-8

import os
import time
import argparse
import traceback

import cv2

import config
from measure.trunk.seg_trunk import segment_trunk_int
from measure.trunk.cal_trunk import Trunk
from measure.calibrator.calibrator_factory import get_calibrator
from util.resize_image import resize_image_with_ratio, recover_coordinate
from util.result import Result, InfoEnum
from util.show_image import *
from util.my_io import *
from util.my_logger import save_log


DEBUG = False
SHOW = False


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure')
    parser.add_argument('-i', '--input',    help='Please provide image list path.')
    parser.add_argument('-o', '--output', help='Please provide output file path.')
    parser.add_argument('-l', '--image_list', help='Please provide image list (use \\n to separate).')
    parser.add_argument('-c', '--calibrator', help='User "laser" or "tag" to calibrate.', default='laser')
    args = parser.parse_args()
    return args


def measure_all(image_path_list):
    """
    为一批图像测算树径
    :param image_path_list: 图像路径列表
    :return: 树径测算结果列表
    """

    if image_path_list is None or len(image_path_list) == 0:
        # 输入不合法
        return {'results': []}

    seg_count = 0           # 计数分割操作的次数
    results_all = []        # 保存所有结果
    image_num = len(image_path_list)

    for i, im_path in enumerate(image_path_list):
        im_id = im_path.split('/')[-1].split('.')[0]

        result = Result()   # 当前图片测量结果
        result.set_image_path(im_path)
        results_all.append(result.get_result())
        time_start = time.time()

        if DEBUG:
            print('-' * 30)
            print('[%d/%d]: %s' % (image_num, i + 1, im_path.split('/')[-1]))

        if not os.path.exists(im_path):
            # 图像不存在
            result.set_info(InfoEnum.IMAGE_NOT_EXIST)
            continue

        im_org = cv2.imread(im_path)                        # 加载图片
        im, resize_ratio = resize_image_with_ratio(im_org)  # 调整尺寸

        # step1: 检测标定物
        calibrator = get_calibrator(im, im_id, DEBUG and SHOW)
        if calibrator is None:
            result.set_info(InfoEnum.CALIBRATOR_DET_FAILED)
            continue

        # 检查站位
        calibrator_ratio = calibrator.get_pixel_scale() / im.shape[0]
        if calibrator_ratio > 0.25:
            result.set_info(InfoEnum.STAND_TOO_CLOSE)
            return result
        
        # 在结果中保存标定物坐标
        # resize坐标 -> 原始坐标
        org_calibrate_pts = []
        for calibrate_pt in calibrator.get_calibrate_points():
            org_calibrate_pt = recover_coordinate(calibrate_pt, resize_ratio)
            org_calibrate_pts.append(org_calibrate_pt)
        result.set_calibrate_points(org_calibrate_pts)

        # step2: 覆盖标定物，避免标定物影响分割
        im_cover = calibrator.cover_calibrator(im)

        if DEBUG:
            visualize_image(im_cover, 'img_cover', im_id=im_id, show=DEBUG and SHOW)

        # 参考标定物切割图片
        # 图片块尺寸由小到大迭代尝试
        crop_params = [2, 4]
        for j, n_crop in enumerate(crop_params):
            im_patch = calibrator.crop_image(im_cover, n_dis_w=n_crop)
            patch_h, patch_w, _ = im_patch.shape

            if DEBUG:
                print('H:%d, W:%d' % (patch_h, patch_w))
                visualize_image(im_patch, 'patch_%d' % n_crop, im_id=im_id, show=DEBUG and SHOW)

            # step4: 分割树干
            if im_patch.shape[0] > config.NET_MAX_WIDTH and im_patch.shape[1] > config.NET_MAX_WIDTH:
                # 待分割的目标尺寸太大
                result.set_info(InfoEnum.TRUNK_TOO_THICK)
                break

            # 交互式分割
            trunk_mask = segment_trunk_int(im_patch, calibrator.positive_pts(), None, im_id=seg_count)
            seg_count += 1

            if DEBUG:
                visualize_image(trunk_mask, 'trunk_mask', im_id=im_id, show=DEBUG and SHOW)

            # step5: 计算树径
            trunk = Trunk(trunk_mask)

            if DEBUG:
                visualize_image(trunk.contour_mask, 'trunk_contour', im_id=im_id, show=DEBUG and SHOW)

            if not trunk.is_seg_succ():
                # 初步估计分割是否成功
                result.set_info(InfoEnum.TRUNK_EDGE_UNCLEAR)
                if DEBUG:
                    print(InfoEnum.TRUNK_EDGE_UNCLEAR)
                continue
            else:
                # 计算实际树径
                RP_ratio = calibrator.RP_ratio()             # 缩放因子
                shot_distance = calibrator.shot_distance()   # 拍摄距离
                trunk_width, seg_conf, patch_trunk_corners = trunk.real_width_v2(shot_distance, RP_ratio)
                if trunk_width > 0:
                    # 置信度：分割置信度 x 标定置信度
                    conf = seg_conf * calibrator.get_conf()
                    time_end = time.time()
                    time_consume = int(time_end - time_start)

                    # 图片块坐标 -> resize图片坐标
                    trunk_left_top = calibrator.recover_coordinate(patch_trunk_corners['left_top'])
                    trunk_right_top = calibrator.recover_coordinate(patch_trunk_corners['right_top'])
                    trunk_left_bottom = calibrator.recover_coordinate(patch_trunk_corners['left_bottom'])
                    trunk_right_bottom = calibrator.recover_coordinate(patch_trunk_corners['right_bottom'])

                    # resize坐标 -> 原图坐标
                    trunk_left_top = recover_coordinate(trunk_left_top, resize_ratio)
                    trunk_left_bottom = recover_coordinate(trunk_left_bottom, resize_ratio)
                    trunk_right_top = recover_coordinate(trunk_right_top, resize_ratio)
                    trunk_right_bottom = recover_coordinate(trunk_right_bottom, resize_ratio)

                    result.set_width(trunk_width)
                    result.set_conf(conf)
                    result.set_trunk_left_top(trunk_left_top)
                    result.set_trunk_right_top(trunk_right_top)
                    result.set_trunk_left_bottom(trunk_left_bottom)
                    result.set_trunk_right_bottom(trunk_right_bottom)
                    result.set_info(InfoEnum.SUCCESS)
                    result.set_time(time_consume)

                    if DEBUG:
                        print('Trunk width: %.2f CM (%.2f).' % (trunk_width / 10.0, conf))
                        pts = org_calibrate_pts + \
                              [trunk_left_top, trunk_left_bottom, trunk_right_top, trunk_right_bottom]
                        show_pts(im_org, pts, im_id=im_id, show=DEBUG and SHOW)
                    break
                else:
                    result.set_info(InfoEnum.TRUNK_EDGE_UNCLEAR)
                    if DEBUG:
                        print('Error is too large.')

    output = {'results': results_all}
    return output


if __name__ == '__main__':

    try:
        args = parse_args()
        input_path = args.input
        output_path = args.output
        raw_list = args.image_list
        calibrator_type = args.calibrator

        if calibrator_type == 'tag':
            config.CALIBRATOR = 'tag'

        if input_path is not None:
            image_list = load_image_list(input_path)
        elif raw_list is not None:
            image_list = parse_image_list(raw_list)
        else:
            image_list = None
        results = measure_all(image_list)
        print_json(results)

        if output_path is not None:
            save_path = args.output
            save_results(results, save_path)
    except:
        if 'args' in dir():
            save_log(args.__str__() + '\n' + traceback.format_exc())
        else:
            save_log(traceback.format_exc())
