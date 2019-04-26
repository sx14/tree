import os
import cv2
import numpy as np
from scipy.io import loadmat

color_tag_dir = 'color_tags'
if not os.path.exists(color_tag_dir):
    os.makedirs(color_tag_dir)

color150 = loadmat('data/color150.mat')['colors']
obj150_path = 'data/object150_info.csv'
with open(obj150_path) as f:
    lines = f.readlines()
    objs = [l.strip().split(',')[-1] for l in lines[1:]]
    objs = [names.split(';')[0] for names in objs]

for i, obj in enumerate(objs):
    color_tag = np.zeros((50, 100, 3))
    color = color150[i]
    color_tag[:, :] = color
    color_tag = color_tag.astype(np.uint8)
    tag_save_path = os.path.join(color_tag_dir, '%d(%s).jpg' % (i+1, obj))
    cv2.imwrite(tag_save_path, color_tag)
