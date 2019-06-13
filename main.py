# coding: utf-8
from det.trunk.seg_trunk import *
from det.trunk.cal_trunk import *
from det.laser.seg_laser_dev import *
from det.laser.laser import *
from util.show_image import *
from util.resize_image import *
import argparse

DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure V1.1')
    parser.add_argument('-i', '--input', help='Please provide image list path.', required=True, type=str)
    parser.add_argument('-o', '--output', help='Please provide output file path.', type=str)
    args = parser.parse_args()
    return args


def measure_tree_width(img_path_list):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    results = []
    count = 0
    for i, im_path in enumerate(sorted(img_path_list)):

        result = {
            'width': -1,
            'conf': -1,
            'info': 'Error is too large.',
            'left_top': None,
            'right_top': None,
            'left_bottom': None,
            'right_bottom': None,
            'laser_top': None,
            'laser_bottom': None
        }

        if DEBUG:
            print('-' * 30)
            print('[%d/%d]: %s' % (len(img_path_list), i+1, im_path.split('/')[-1]))

        im = cv2.imread(im_path)
        im = resize_image(im)

        # step1 获得激光点对位置，激光点mask，激光mask，激光得分
        pt_pair, pt_mask, laser_mask, pt_conf = get_laser_points(im)

        if DEBUG:
            show_images([im, laser_mask, pt_mask])

        if len(pt_pair) != 2:
            result['width'] = -1
            result['conf'] = -1
            result['info'] = 'Laser points detection failed.'
            if DEBUG:
                print('Laser point pair detection failed.')
            continue
        else:
            if DEBUG:
                print('Laser point pair detection success.')

        # 在结果中保存激光点坐标
        laser = Laser(pt_pair, pt_mask, laser_mask)
        laser_top_pt = laser.get_top_pt()
        laser_bottom_pt = laser.get_bottom_pt()
        result['laser_top'] = laser_top_pt
        result['laser_bottom'] = laser_bottom_pt

        # step2: 覆盖激光区域
        # TODO: 设备修改后，不需要覆盖激光线
        im_cover = laser.cover_laser(im)

        if DEBUG:
            show_image(im_cover, 'cover')

        crop_params = [2, 3, 4]
        for j, n_crop in enumerate(crop_params):

            # step3: 根据激光点距离，切割图片
            im_patch = laser.crop_image(im_cover, n_dis_w=n_crop)
            patch_h, patch_w, _ = im_patch.shape

            if DEBUG:
                print('H:%d, W:%d' % (patch_h, patch_w))
                show_image(im_patch, 'patch')

            # step4: 分割高置信度背景区域（叶子）
            # leaf_mask = segment_leaf(im_patch)
            # bg_mask = leaf_mask
            bg_mask = None

            # if DEBUG:
                # show_image(leaf_mask, 'leaf')

            # step5: 分割树干
            if max(im_patch.shape[0], im_patch.shape[1]) > NET_MAX_WIDTH:
                if DEBUG:
                    print('[ ERROR ] Image is too large to segmented. Move further away from target.')
                result['width'] = -1,
                result['conf'] = -1,
                result['info'] = 'Stand too close to tree.'
                break

            show_img, trunk_mask = segment_trunk_int(im_patch, laser.positive_pts(), bg_mask, im_id=count)
            count += 1

            if DEBUG:
                show_images([show_img, trunk_mask], 'segment')
                # cv2.imwrite('im_crop.jpg', im)
                # cv2.imwrite('trunk_mask.png', trunk_mask)

            # step6: 计算树径
            trunk = Trunk(trunk_mask)

            if DEBUG:
                # cv2.imwrite('trunk_contour.png', trunk.contour_mask)
                show_images([im, trunk.trunk_mask, trunk.contour_mask], 'trunk')

            if not trunk.is_seg_succ():
                result['width'] = -1,
                result['conf'] = -1,
                result['info'] = 'Trunk edges are not clear.'

                if DEBUG:
                    print('Trunk edges are not clear.')
                continue
            else:
                # 计算拍摄距离
                RP_ratio = laser.RP_ratio()
                shot_distance = laser.shot_distance()
                trunk_width, seg_conf, patch_trunk_corners = trunk.real_width_v2(shot_distance, RP_ratio)
                if trunk_width > 0:
                    conf = seg_conf * pt_conf

                    # 图片分块坐标 -> 原始图片坐标
                    trunk_left_top = laser.recover_coordinate(patch_trunk_corners['left_top'], n_dis_w=n_crop)
                    trunk_right_top = laser.recover_coordinate(patch_trunk_corners['right_top'], n_dis_w=n_crop)
                    trunk_left_bottom = laser.recover_coordinate(patch_trunk_corners['left_bottom'], n_dis_w=n_crop)
                    trunk_right_bottom = laser.recover_coordinate(patch_trunk_corners['right_bottom'], n_dis_w=n_crop)

                    result['width'] = trunk_width
                    result['conf'] = conf
                    result['info'] = 'success'
                    result['left_top'] = trunk_left_top
                    result['right_top'] = trunk_right_top
                    result['left_bottom'] = trunk_left_bottom
                    result['right_bottom'] = trunk_right_bottom

                    if DEBUG:
                        print('Trunk width: %.2f CM (%.2f).' % (trunk_width / 10.0, conf))

                    pts = [laser_bottom_pt, laser_top_pt,
                           trunk_left_top, trunk_left_bottom,
                           trunk_right_top, trunk_right_bottom]

                    im_plt = np.stack((im[:, :, 2], im[:, :, 1], im[:, :, 0]), axis=2)

                    if DEBUG:
                        show_pts(im_plt, pts)

                    if conf > 0.1:
                        break
                else:
                    if DEBUG:
                        print('Error is too large.')

        results.append(result)
    output = {'results': results}
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

    result = measure_tree_width(image_list)
    print(result)

    # save_path = args.output
    # save_results(output, save_path)
