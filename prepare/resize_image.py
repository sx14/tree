import os
import cv2
import numpy as np
from util.show_image import *

org_im_root = '../data/tree_tag/'
tmp_im_root = '../data/tree_tag1/'
for img_id in os.listdir(org_im_root):
    im_path = org_im_root + img_id
    im = cv2.imread(im_path)
    # resize
    im_h, im_w, _ = im.shape
    resize_ratio = 800.0 / im_h
    im_h = int(im_h * resize_ratio)
    im_w = int(im_w * resize_ratio)
    im_crop = cv2.resize(im, (im_w, im_h))

    im_output_path = tmp_im_root+img_id
    cv2.imwrite(im_output_path, im_crop.astype(np.uint8))
