# coding: utf-8
import os
import argparse

import cv2

import config
from det.trunk.seg_trunk import segment_trunk_int
from det.trunk.cal_trunk import Trunk

if config.CALIBRATOR == 'laser':
    from det.calibrator.laser.seg_laser import get_laser_points as get_calibrator_points
    from det.calibrator.laser.laser import Laser as Calibrator
else:
    from det.calibrator.tag.seg_tag import segment_tag as get_calibrator_points
    from det.calibrator.tag.tag import BlueTag as Calibrator

from util.resize_image import resize_image_with_ratio, recover_coordinate
from util.result import Result, InfoEnum
from util.show_image import *
from util.my_io import *


DEBUG = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure V1.1')
    parser.add_argument('-i', '--input',    help='Please provide image list path.', default='None')
    parser.add_argument('-o', '--output', help='Please provide output file path.', default='None')
    parser.add_argument('-l', '--image_list', help='Please provide image list (use \\n to separate).', default='None')
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
        if DEBUG:
            print('Bad image path list.')
        return {'results': []}

    seg_count = 0           # 计数分割操作的次数
    all_results = []        # 保存所有结果
    image_num = len(image_path_list)

    for i, im_path in enumerate(image_path_list):
        im_id = im_path.split('/')[-1][:-4]

        result = Result()   # 当前图片的分割结果，初始化为失败
        result.set_image_path(im_path)
        all_results.append(result.get_result())

        if DEBUG:
            print('-' * 30)
            print('[%d/%d]: %s' % (image_num, i + 1, im_path.split('/')[-1]))

        if not os.path.exists(im_path):
            # 图像不存在
            result.set_info(InfoEnum.IMAGE_NOT_EXIST)
            continue

        im_org = cv2.imread(im_path)                        # 加载图片
        im, resize_ratio = resize_image_with_ratio(im_org)  # 调整尺寸

        # step1: 获得激光点对位置，激光点mask，激光mask，激光得分
        pt_pair, pt_mask, laser_mask, pt_conf = get_calibrator_points(im, False)

        if DEBUG:
            visualize_image(im, name='img', im_id=im_id)
            visualize_image(pt_mask, name='pt', im_id=im_id)

        if len(pt_pair) != 2:
            result.set_info(InfoEnum.LASER_DET_FAILED)
            if DEBUG:
                print(InfoEnum.LASER_DET_FAILED)
            continue
        else:
            if DEBUG:
                print('Laser point detection success.')

        calibrator = Calibrator(pt_pair, pt_mask, laser_mask)

        # 在结果中保存激光点坐标
        # resize坐标 -> 原始坐标
        laser_top_pt = recover_coordinate(calibrator.get_top_pt(), resize_ratio)
        laser_bottom_pt = recover_coordinate(calibrator.get_bottom_pt(), resize_ratio)
        result.set_laser_top(laser_top_pt)
        result.set_laser_bottom(laser_bottom_pt)

        # step2: 覆盖激光区域
        # TODO: 设备修改后，不需要覆盖激光线
        im_cover = calibrator.cover_calibrator(im)

        if DEBUG:
            visualize_image(im_cover, 'img_cover', im_id=im_id)

        # 根据激光点距离，切割图片，缩小分割范围
        # 图片块尺寸由小到大迭代尝试
        crop_params = [2, 3, 4]
        for j, n_crop in enumerate(crop_params):
            im_patch = calibrator.crop_image(im_cover, n_dis_w=n_crop)
            patch_h, patch_w, _ = im_patch.shape

            if DEBUG:
                print('H:%d, W:%d' % (patch_h, patch_w))
                visualize_image(im_patch, 'patch_%d' % n_crop, im_id=im_id)

            # step4: 分割树干
            if max(im_patch.shape[0], im_patch.shape[1]) > config.NET_MAX_WIDTH:
                # 待分割的目标图像块尺寸太大
                result.set_info(InfoEnum.STAND_TOO_CLOSE)
                if DEBUG:
                    print('[ ERROR ] Trunk is too thick.')
                break

            # 交互式分割
            trunk_mask = segment_trunk_int(im_patch, calibrator.positive_pts(), None, im_id=seg_count)
            seg_count += 1

            if DEBUG:
                # show_images([im, trunk_mask], 'segment')
                visualize_image(trunk_mask, 'trunk_mask', im_id=im_id)

            # step5: 计算树径
            trunk = Trunk(trunk_mask)

            if DEBUG:
                # show_images([im, trunk.trunk_mask, trunk.contour_mask], 'trunk')
                visualize_image(trunk.contour_mask, 'trunk_contour', im_id=im_id)

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
                    # 置信度：分割置信度 x 激光点置信度
                    conf = seg_conf * pt_conf
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

                    if DEBUG:
                        print('Trunk width: %.2f CM (%.2f).' % (trunk_width / 10.0, conf))
                        pts = [laser_bottom_pt, laser_top_pt,
                               trunk_left_top, trunk_left_bottom,
                               trunk_right_top, trunk_right_bottom]
                        show_pts(im_org, pts, im_id=im_id)
                    break
                else:
                    result.set_info(InfoEnum.TRUNK_EDGE_UNCLEAR)
                    if DEBUG:
                        print('Error is too large.')

    output = {'results': all_results}
    return output


if __name__ == '__main__':
    args = parse_args()
    input_path = args.input
    output_path = args.output
    raw_list = args.image_list

    if input_path != 'None':
        image_list = load_image_list(input_path)
    elif raw_list != 'None':
        image_list = parse_image_list(raw_list)
    else:
        image_list = None

    results = measure_all(image_list)
    print_json(results)

    if output_path != 'None':
        save_path = args.output
        save_results(results, save_path)
