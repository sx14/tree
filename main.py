# coding: utf-8
from det.trunk.seg_trunk import *
from det.trunk.cal_trunk import *
from det.tag.seg_tag import *
from det.tag.cal_tag import *
from util.show_image import *
from util.resize_image import *

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # img_root = 'data/tree_tag_ps'
    img_root = 'data/tree_tag_test/2'
    count = 0
    for i, img_id in enumerate(os.listdir(img_root)):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('%d: Img %s' % (i, img_id))
        im_path = os.path.join(img_root, img_id)
        im_org = cv2.imread(im_path)
        im_org = resize_image(im_org)

        # step1: segment blue tag at image center
        tag_mask_org = segment_tag(im_org)
        if tag_mask_org is None:
            print('Tag detection failed.')
            continue

        # check camera angle
        tag = BlueTag(tag_mask_org)
        # cv2.imwrite('im_org.jpg', im_org)
        # cv2.imwrite('tag_mask.png', tag.mask)
        # cv2.imwrite('tag_contour.png', tag.show_lines())
        # show_images('', [im_org, tag.mask, tag.show_lines()])

        if not tag.is_edge_det_succ():
            print('Tag detection failed.')
            continue
        if tag.check_parallel() != tag.SUCCESS:
            print('Vertical angle is too large.')
            continue
        if tag.check_perpendicular() != tag.SUCCESS:
            print('Horizontal angle is too large.')
            continue

        RP_ratio = tag.RP_ratio()
        shot_distance = tag.shot_distance()
        if RP_ratio is None or shot_distance is None:
            print('[WARNING] Internal error: Tag calculation failed.')
            continue

        crop_params = [2, 4, 6]
        for j, n_crop in enumerate(crop_params):
            # step2: crop image patch around tag
            im, tag_mask, _ = crop_image(im_org, tag_mask_org, n_tag_w=n_crop)

            # step3: remove blue tag from image
            im = remove_tag1(im, tag_mask)

            # step4: segment definitive background part
            leaf_mask = segment_leaf(im)
            pr_bg_mask = leaf_mask

            # step5: segment tree trunk
            if max(im.shape[0], im.shape[1]) > NET_MAX_WIDTH:
                print('Image is too large to segmented. Move further away from target.')
                break

            show_img, trunk_mask = segment_trunk_int(im, tag_mask, pr_bg_mask, im_id=count)
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


