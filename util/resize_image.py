import cv2
from config import IMG_MAX_HEIGHT


def resize_image(im):
    # image is vertical.
    # default size: H=4000, W=3000
    im_h, im_w, _ = im.shape

    if IMG_MAX_HEIGHT < im_h:
        resize_ratio = IMG_MAX_HEIGHT * 1.0 / im_h
        im_h = int(im_h * resize_ratio)
        im_w = int(im_w * resize_ratio)
        im_resize = cv2.resize(im, (im_w, im_h))
        return im_resize
    else:
        return im


