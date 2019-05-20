import cv2


def show_image(im, win_name='show'):
    im_h = im.shape[0]
    im_w = im.shape[1]
    ratio = 800.0 / im_h

    cv2.namedWindow(win_name, 0)
    cv2.resizeWindow(win_name, int(im_w * 1.0 * ratio), int(im_h * 1.0 * ratio))
    cv2.moveWindow(win_name, 200, 200)
    cv2.imshow(win_name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images(ims, name='show'):
    left_top_x = 0

    for i, im in enumerate(ims):
        win_name = 'name%s[%d]' % (name, i+1)
        im_h = im.shape[0]
        im_w = im.shape[1]
        ratio = 800.0 / im_h

        cv2.namedWindow(win_name, 0)
        cv2.resizeWindow(win_name, int(im_w * 1.0 * ratio), int(im_h * 1.0 * ratio))
        cv2.moveWindow(win_name, left_top_x, 0)
        cv2.imshow(win_name, im)
        left_top_x += int(im_w * 1.0 * ratio)

    cv2.waitKey(0)
    cv2.destroyAllWindows()