# coding: utf-8
import cv2
import numpy as np
from util.show_image import *


# 色相， 色度， 明度
red1_fr = [158,50,190]
red1_to = [180,255,255]


red2_fr = [0,50,190]
red2_to = [6,255,255]
red2_test = [0, 100, 173]


patch = np.zeros((500,500,3)).astype(np.uint8)
patch[:,:] = red2_test
im = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)
visualize_image(im, '')


