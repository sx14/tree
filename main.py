# coding: utf-8
import os
from det.trunk.seg_trunk import *
from det.trunk.cal_trunk import *
from det.laser.seg_laser_dev import *
from det.laser.laser import *
from util.show_image import *
from util.resize_image import *
from util.result import Result, InfoEnum
import argparse

DEBUG = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure V1.1')
    parser.add_argument('-i', '--input', help='Please provide image list path.', required=True, type=str)
    parser.add_argument('-o', '--output', help='Please provide output file path.', type=str)
    args = parser.parse_args()
    return args


def measure_tree_width(image_path_list):
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

        im = cv2.imread(im_path)    # 加载图片
        im = resize_image(im)       # 调整尺寸

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

        # 在结果中保存激光点坐标
        laser = Laser(pt_pair, pt_mask, laser_mask)
        laser_top_pt = laser.get_top_pt()
        laser_bottom_pt = laser.get_bottom_pt()
        result.set_laser_top(laser_top_pt)
        result.set_laser_bottom(laser_bottom_pt)

        # step2: 覆盖激光区域
        # TODO: 设备修改后，不需要覆盖激光线
        im_cover = laser.cover_laser(im)

        if DEBUG:
            show_image(im_cover, 'cover')

        # 根据激光点距离，切割图片，缩小分割范围
        # 图片块尺寸由小到大
        crop_params = [2, 3, 4]
        for j, n_crop in enumerate(crop_params):
            im_patch = laser.crop_image(im_cover, n_dis_w=n_crop)
            patch_h, patch_w, _ = im_patch.shape

            if DEBUG:
                print('H:%d, W:%d' % (patch_h, patch_w))
                show_image(im_patch, 'patch')

            # step4: 分割树干
            if max(im_patch.shape[0], im_patch.shape[1]) > NET_MAX_WIDTH:
                result.set_info(InfoEnum.STAND_TOO_CLOSE)
                if DEBUG:
                    print('[ ERROR ] Image is too large to segmented. Move further away from target.')
                break

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
                result.set_info(InfoEnum.TRUNK_EDGE_UNCLEAR)
                if DEBUG:
                    print(InfoEnum.TRUNK_EDGE_UNCLEAR)
                continue
            else:
                # 计算拍摄距离
                RP_ratio = laser.RP_ratio()
                shot_distance = laser.shot_distance()
                trunk_width, seg_conf, patch_trunk_corners = trunk.real_width_v2(shot_distance, RP_ratio)
                if trunk_width > 0:
                    conf = seg_conf * pt_conf
                    # 图片块坐标 -> 原始图片坐标
                    trunk_left_top = laser.recover_coordinate(patch_trunk_corners['left_top'], n_dis_w=n_crop)
                    trunk_right_top = laser.recover_coordinate(patch_trunk_corners['right_top'], n_dis_w=n_crop)
                    trunk_left_bottom = laser.recover_coordinate(patch_trunk_corners['left_bottom'], n_dis_w=n_crop)
                    trunk_right_bottom = laser.recover_coordinate(patch_trunk_corners['right_bottom'], n_dis_w=n_crop)

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
                        im_plt = np.stack((im[:, :, 2], im[:, :, 1], im[:, :, 0]), axis=2)
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


def load_image_list(image_list_path):
    if os.path.exists(image_list_path):
        with open(list_path) as f:
            image_paths = [l.strip() for l in f.readlines()]
        return image_paths
    else:
        return None


def save_results(results, save_path):
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    args = parse_args()
    list_path = args.input
    image_list = load_image_list(list_path)

    output = measure_tree_width(image_list)
    print(output)

    save_path = args.output
    save_results(output, save_path)
