# coding: utf-8
from det.trunk.seg_trunk import *
from det.trunk.cal_trunk import *
from det.laser.seg_laser import *
from det.laser.cal_laser import *
from util.show_image import *
from util.resize_image import *

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    img_root = 'data/tree'

    count = 0
    for i, img_id in enumerate(os.listdir(img_root)):

        print('>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('%d: Img %s' % (i, img_id))
        im_path = os.path.join(img_root, img_id)
        im = cv2.imread(im_path)

        # step1: segment blue tag at image center
        pt_pair, pt_mask, laser_mask = segment_laser_pts(im)

        # lm = laser_mask.copy()
        # lm[lm > 0] = 255
        # pm = pt_mask.copy()
        # pm[pm > 0] = 255
        # if len(pt_pair) == 0:
        #     show_images([im, lm, pm], name=str(i))
        # else:
        #     show_image(pm)
        if len(pt_pair) != 2:
            print('[ ERROR ] Laser point pair detection failed.')\

        laser = Laser(pt_pair, pt_mask, laser_mask)

        # 计算拍摄距离
        RP_ratio = laser.RP_ratio()
        shot_distance = laser.shot_distance()

        crop_params = [2, 3, 4]
        for j, n_crop in enumerate(crop_params):
            # step2: 覆盖激光区域
            im_cover = laser.cover_laser(im)
            # show_image(im_cover)

            # step3: 根据激光点距离，切割图片
            im_patch = laser.crop_image(im_cover, n_crop)

            # step4: 分割高置信度背景区域（叶子）
            leaf_mask = segment_leaf(im_patch)
            pr_bg_mask = leaf_mask

            # step5: segment tree trunk
            if max(im_patch.shape[0], im_patch.shape[1]) > NET_MAX_WIDTH:
                print('[ ERROR ] Image is too large to segmented. Move further away from target.')
                break

            show_img, trunk_mask = segment_trunk_int(im_patch, laser.positive_pts(), pr_bg_mask, im_id=count)
            count += 1
            # show_images(img_id, [show_img, tag_mask, trunk_mask])
            # cv2.imwrite('im_crop.jpg', im)
            # cv2.imwrite('trunk_mask.png', trunk_mask)


            # step6: calculate
            trunk = Trunk(trunk_mask)
            # cv2.imwrite('trunk_contour.png', trunk.contour_mask)
            # show_images('', [im, trunk.trunk_mask, trunk.contour_mask])
            if not trunk.is_seg_succ():
                print('Trunk segmentation failed.')
                continue
            else:
                trunk_width = trunk.real_width_v2(shot_distance, RP_ratio)
                if trunk_width is not None:
                    print('Trunk width: %.2f CM.' % (trunk_width / 10.0))
                    break
                else:
                    print('Measure point or background or crop.')


