import os
import cv2
import numpy as np


tree_pkgs = ['tree', 'tree_m']

for tree_pkg in tree_pkgs:
    tree_pkg_path = os.path.join('../data/%s' % tree_pkg)
    tree_img_ids = os.listdir(tree_pkg_path)

    flipped_tree_pkg_path = os.path.join('../data/%s_flipped' % tree_pkg)
    if not os.path.exists(flipped_tree_pkg_path):
        os.mkdir(flipped_tree_pkg_path)

    for tree_id in tree_img_ids:
        print('[%s] %s' % (tree_pkg, tree_id))

        tree_path = os.path.join(tree_pkg_path, tree_id)
        im = cv2.imread(tree_path)
        im_flipped = np.zeros(im.shape).astype(np.uint8)
        im_w = im.shape[1]
        for c in range(im_w):
            im_flipped[:, c, :] = im[:, im_w-1-c, :]

        im_flipped_path = os.path.join(flipped_tree_pkg_path, tree_id)
        cv2.imwrite(im_flipped_path, im_flipped)

