# coding: utf-8
import os
from det.laser.seg_laser_dev import *
from det.laser.laser import *
from util.show_image import *
from util.resize_image import *
from config import IMG_MAX_HEIGHT

if __name__ == '__main__':
    """
    实验确定镜头焦距FOCAL_LENGTH
    单位是pixel
    """

    img_root = '../data/laser'

    focal_dis_arr = []

    print('Image height(resized): %d' % IMG_MAX_HEIGHT)
    print('dis mm: focal pix (conf)')
    print('-' * 30)
    for i, img_id in enumerate(sorted(os.listdir(img_root))):
        im_path = os.path.join(img_root, img_id)
        im = cv2.imread(im_path)
        im = resize_image(im)

        pt_pair, pt_mask, laser_mask, pt_score = segment_laser_pts(im, calibrate=True)
        # show_images([im, laser_mask, pt_mask])

        if len(pt_pair) == 0:
            print('[WARNING] Laser point pair detection failed.')
            continue

        laser = Laser(pt_pair, pt_mask, laser_mask)

        real_dis = laser.POINT_DISTANCE
        pixel_dis = laser.point_pixel_dis()

        shot_dis = int(img_id.split('.')[0]) * 10.0 + 30

        focal_dis = shot_dis * (pixel_dis / real_dis)
        print('%d mm: %.2f pix (%.2f)' % (shot_dis, focal_dis, pt_score))
        focal_dis_arr.append(focal_dis)

    print('-' * 30)
    focal_dis_arr = np.array(focal_dis_arr)
    print('VAR: %.2f' % (focal_dis_arr.var()))

    focal_dis_avg = sum(focal_dis_arr) / len(focal_dis_arr)
    print('FOCAL_LENGTH=%.4f pix' % focal_dis_avg)




