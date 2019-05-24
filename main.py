# coding: utf-8
from det.trunk.seg_trunk import *
from det.trunk.cal_trunk import *
from det.laser.seg_laser import *
from det.laser.laser import *
from util.show_image import *
from util.resize_image import *


def measure_tree_width(img_path_list):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # width, conf, info
    results = []

    count = 0
    for i, im_path in enumerate(sorted(img_path_list)):

        print('-' * 30)
        print('%d: Img %s' % (i, im_path))
        im = cv2.imread(im_path)
        im = resize_image(im)

        # step1 获得激光点对位置，激光点mask，激光mask，激光得分
        pt_pair, pt_mask, laser_mask, pt_conf = segment_laser_pts(im, calibrate=False)
        # show_images([im, laser_mask, pt_mask])

        if len(pt_pair) != 2:
            results.append([-1, -1, 'Laser points detection failed.'])
            continue
        laser = Laser(pt_pair, pt_mask, laser_mask)

        # step2: 覆盖激光区域
        # TODO: 设备修改后，不需要覆盖激光线
        im_cover = laser.cover_laser(im)
        # show_image(im_cover, 'cover')

        result = [-1, -1, 'Error is too large.']
        crop_params = [2, 3, 4]
        for j, n_crop in enumerate(crop_params):

            # step3: 根据激光点距离，切割图片
            im_patch = laser.crop_image(im_cover, n_dis_w=n_crop)
            patch_h, patch_w, _ = im_patch.shape
            print('H:%d, W:%d' % (patch_h, patch_w))
            # show_image(im_patch, 'patch')

            # step4: 分割高置信度背景区域（叶子）
            leaf_mask = segment_leaf(im_patch)
            bg_mask = leaf_mask
            # show_image(leaf_mask, 'leaf')

            # step5: 分割树干
            if max(im_patch.shape[0], im_patch.shape[1]) > NET_MAX_WIDTH:
                print('[ ERROR ] Image is too large to segmented. Move further away from target.')
                results.append([-1, 0.0, 'Trunk segmentation failed.'])
                break

            show_img, trunk_mask = segment_trunk_int(im_patch, laser.positive_pts(), bg_mask, im_id=count)
            # show_images([show_img, trunk_mask], 'segment')
            count += 1
            # show_images(img_id, [show_img, tag_mask, trunk_mask])
            # cv2.imwrite('im_crop.jpg', im)
            # cv2.imwrite('trunk_mask.png', trunk_mask)

            # step6: 计算树径
            trunk = Trunk(trunk_mask)
            # cv2.imwrite('trunk_contour.png', trunk.contour_mask)
            # show_images([im, trunk.trunk_mask, trunk.contour_mask], 'trunk')

            if not trunk.is_seg_succ():
                print('Trunk segmentation failed.')
                continue
            else:
                # 计算拍摄距离
                RP_ratio = laser.RP_ratio()
                shot_distance = laser.shot_distance()
                trunk_width, seg_conf = trunk.real_width_v2(shot_distance, RP_ratio)
                if trunk_width > 0:
                    conf = seg_conf * pt_conf
                    result = [trunk_width, conf, 'success']
                    print('Trunk width: %.2f CM (%.2f).' % (trunk_width / 10.0, conf))
                    if conf > 0.7:
                        break
                else:
                    print('Error is too large.')
        results.append(result)

    return results


if __name__ == '__main__':

    img_root = 'data/tree_m'
    img_path_list = [os.path.join(img_root, img_id) for img_id in sorted(os.listdir(img_root))]
    results = measure_tree_width(img_path_list)

    # TODO: save results






