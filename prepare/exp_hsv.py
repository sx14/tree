# coding: utf-8
import cv2
import numpy as np
from util.show_image import *


# 色相， 色度， 明度
# red1_fr = [158,50,190]
# red1_to = [180,255,255]
#
#
# red2_fr = [0,50,190]
# red2_to = [6,255,255]
# red2_test = [255,0,0]
#
#
# patch = np.zeros((500,500,3)).astype(np.uint8)
# patch[:,:] = red2_test
# im = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)
# show_image(im, '')


def show_hsv_value(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    def print_hsv(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(im_hsv[y,x])

    # 创建图像与窗口并将窗口与回调函数绑定
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', print_hsv)
    while (1):
        cv2.imshow('image', im)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_list_path = '../image_list.txt'
    with open(image_list_path) as f:
        image_list = f.readlines()
        image_list = [l.strip() for l in image_list]
    for image_path in image_list:
        im = cv2.imread(image_path)
        im = cv2.resize(im, (600, 800))
        show_hsv_value(im)