# coding: utf-8
import os
import argparse

import cv2
import numpy as np

from det.trunk.seg_trunk import segment_trunk_int
from det.trunk.cal_trunk import Trunk
from det.laser.seg_laser_dev import get_laser_points, NET_MAX_WIDTH
from det.laser.laser import Laser
from util.resize_image import resize_image_with_ratio, recover_coordinate
from util.result import Result, InfoEnum
from util.show_image import *
from util.my_io import *


DEBUG = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure V1.1')
    parser.add_argument('-i', '--input', help='Please provide image list path.', required=True, type=str)
    parser.add_argument('-o', '--output', help='Please provide output file path.', type=str)
    args = parser.parse_args()
    return args


def measure_all(image_path_list):
    """
    为一批图像测算树径
    :param image_path_list: 图像路径列表
    :return: 树径测算结果列表
    """

    if image_path_list is None or len(image_path_list) == 0:
        # 输入为空
        return []

    seg_count = 0           # 计数分割操作的次数
    all_results = []        # 保存所有结果
    image_num = len(image_path_list)

    for i, im_path in enumerate(image_path_list):
        result = Result()   # 保存当前图片的分割结果，初始化为失败

        if DEBUG:
            print('-' * 30)
            print('[%d/%d]: %s' % (image_num, i + 1, im_path.split('/')[-1]))

        if not os.path.exists(im_path):
            # 图像不存在
            result.set_info(InfoEnum.IMAGE_NOT_EXIST)
            continue

        im_org = cv2.imread(im_path)    # 加载图片
        im, resize_ratio = resize_image_with_ratio(im_org)       # 调整尺寸

        # step1: 获得激光点对位置，激光点mask，激光mask，激光得分
        pt_pair, pt_mask, laser_mask, pt_conf = get_laser_points(im, DEBUG)

        if DEBUG:
            show_images([im, laser_mask, pt_mask])
            pass

        if len(pt_pair) != 2:
            result.set_info(InfoEnum.LASER_DET_FAILED)
            if DEBUG:
                print(InfoEnum.LASER_DET_FAILED)
            continue
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

        # step2: 覆盖激光区域
        # TODO: 设备修改后，不需要覆盖激光线
        im_cover = laser.cover_laser(im)

        if DEBUG:
            show_image(im_cover, 'cover')

        # 根据激光点距离，切割图片，缩小分割范围
        # 图片块尺寸由小到大迭代尝试
        crop_params = [2, 3, 4]
        for j, n_crop in enumerate(crop_params):
            im_patch = laser.crop_image(im_cover, n_dis_w=n_crop)
            patch_h, patch_w, _ = im_patch.shape

            if DEBUG:
                print('H:%d, W:%d' % (patch_h, patch_w))
                show_image(im_patch, 'patch')

            # step4: 分割树干
            if max(im_patch.shape[0], im_patch.shape[1]) > NET_MAX_WIDTH:
                # 待分割的目标图像块尺寸太大
                result.set_info(InfoEnum.STAND_TOO_CLOSE)
                if DEBUG:
                    print('[ ERROR ] Image is too large to segmented. Move further away from target.')
                break

            # 交互式分割
            show_img, trunk_mask = segment_trunk_int(im_patch, laser.positive_pts(), None, im_id=seg_count)
            seg_count += 1

            if DEBUG:
                show_images([show_img, trunk_mask], 'segment')
                # cv2.imwrite('im_crop.jpg', im)
                # cv2.imwrite('trunk_mask.png', trunk_mask)

            # step5: 计算树径
            trunk = Trunk(trunk_mask)

            if DEBUG:
                show_images([im, trunk.trunk_mask, trunk.contour_mask], 'trunk')
                # cv2.imwrite('trunk_contour.png', trunk.contour_mask)

            if not trunk.is_seg_succ():
                # 初步估计分割是否成功
                result.set_info(InfoEnum.TRUNK_EDGE_UNCLEAR)
                if DEBUG:
                    print(InfoEnum.TRUNK_EDGE_UNCLEAR)
                continue
            else:
                # 计算实际树径
                RP_ratio = laser.RP_ratio()             # 缩放因子
                shot_distance = laser.shot_distance()   # 拍摄距离
                trunk_width, seg_conf, patch_trunk_corners = trunk.real_width_v2(shot_distance, RP_ratio)
                if trunk_width > 0:
                    # 置信度：分割置信度 x 激光点置信度
                    conf = seg_conf * pt_conf
                    # 图片块坐标 -> resize图片坐标
                    trunk_left_top = laser.recover_coordinate(patch_trunk_corners['left_top'], n_dis_w=n_crop)
                    trunk_right_top = laser.recover_coordinate(patch_trunk_corners['right_top'], n_dis_w=n_crop)
                    trunk_left_bottom = laser.recover_coordinate(patch_trunk_corners['left_bottom'], n_dis_w=n_crop)
                    trunk_right_bottom = laser.recover_coordinate(patch_trunk_corners['right_bottom'], n_dis_w=n_crop)

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

                    if True:
                        print('Trunk width: %.2f CM (%.2f).' % (trunk_width / 10.0, conf))
                        pts = [laser_bottom_pt, laser_top_pt,
                               trunk_left_top, trunk_left_bottom,
                               trunk_right_top, trunk_right_bottom]
                        im_plt = np.stack((im_org[:, :, 2], im_org[:, :, 1], im_org[:, :, 0]), axis=2)
                        show_pts(im_plt, pts)

                    if conf > 0.1:
                        break
                else:
                    result.set_info(InfoEnum.TRUNK_EDGE_UNCLEAR)
                    if DEBUG:
                        print('Error is too large.')

        all_results.append(result.get_result())
    output = {'results': all_results}
    return output


if __name__ == '__main__':
    args = parse_args()
    list_path = args.input
    image_list = load_image_list(list_path)
    results = measure_all(image_list)
    print_json(results)

    # save_path = args.output
    # save_results(results, save_path)
