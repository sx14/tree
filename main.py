# coding: utf-8
import os
import cv2
from det.tag.seg_tag import *
from det.trunk.seg_trunk import *
from det.common.det_edge_hof import *
from util.show_image import *

if __name__ == '__main__':

    img_root = 'data/tree_tag'

    for i, img_id in enumerate(os.listdir(img_root)):
        im_path = os.path.join(img_root, img_id)
        im = cv2.imread(im_path)

        # step1: segment blue tag at image center
        tag_mask = segment_tag(im)
        # show_images(img_id, [im, tag_mask])

        # step2: crop image patch around tag
        im, tag_mask, org_tag_box = crop_image(im, tag_mask)
        # tag_box_path = os.path.join('data', 'tree_tag_box', img_id.split('.')[0]+'.png')
        # cv2.imwrite(tag_box_path, org_tag_box)

        # step3: remove blue tag from image
        im = remove_tag1(im, tag_mask)
        # show_images(img_id, [im])

        # step4: segment definitive background part
        leaf_mask = segment_leaf(im)
        cloth_mask = segment_cloth(im)
        pr_bg_mask = leaf_mask | cloth_mask
        # show_images(img_id, [im, leaf_mask, cloth_mask])

        show_img, trunk_mask = segment_trunk_int(im, tag_mask, pr_bg_mask, im_id=i)
        # show_img, trunk_mask = segment_trunk_thr(im, pr_bg_mask)
        show_images(img_id, [im, show_img, trunk_mask])
